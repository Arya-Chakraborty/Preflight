"""The Preflight gateway: analyze -> decide -> act -> log -> remember.

Robustness contract: any failure in analysis, memory, assembly, or policy
degrades to a raw passthrough call (A5) with the error recorded. The gateway
must never block or corrupt a request.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import random
import time
import uuid
from collections.abc import AsyncIterator
from typing import Any

from preflight import tokens
from preflight.analyzer.embeddings import build_embedder, cosine
from preflight.analyzer.features import Features, base_features
from preflight.assembler.assembler import Assembler, Candidate
from preflight.assembler.grounding import GroundingStore
from preflight.config import Settings
from preflight.costs.estimators import load_estimators
from preflight.costs.model import CandidateStats, CostModel
from preflight.costs.prices import provider_of
from preflight.ledger.ledger import LedgerPrediction, PrefixLedger
from preflight.memory.store import MemoryStore, conversation_hash
from preflight.outcomes.logger import Outcome, OutcomeLogger
from preflight.policy.engine import choose, feasible_actions

log = logging.getLogger("preflight")

# Request keys forwarded verbatim to the provider.
_PASSTHROUGH_KEYS = (
    "temperature", "top_p", "max_tokens", "max_completion_tokens", "stop",
    "presence_penalty", "frequency_penalty", "tools", "tool_choice",
    "response_format", "seed", "user", "n",
)


class Gateway:
    def __init__(self, settings: Settings):
        settings.ensure_dirs()
        self.settings = settings
        self.embedder = build_embedder(
            settings.embedder, settings.embedding_model, settings.hashing_dim
        )
        self.memory = MemoryStore(settings.data_dir, self.embedder)
        self.ledger = PrefixLedger(settings.data_dir)
        self.grounding = GroundingStore(settings)
        self.logger = OutcomeLogger(settings.data_dir)
        outlen, pfail = load_estimators(settings)
        self.outlen, self.pfail = outlen, pfail
        self.cost_model = CostModel(settings, outlen, pfail)
        self.assembler = Assembler(settings)
        self.rng = random.Random()
        self._last_query: dict[str, tuple[str, str]] = {}  # session -> (query, request_id)

    # ------------------------------------------------------------ entrypoints

    async def handle(self, payload: dict, session_id: str | None = None) -> dict:
        """Non-streaming request. Returns an OpenAI-format response dict."""
        ctx = self._preflight(payload, session_id)
        if ctx["decision"].action == "A1":
            return self._finish_cached(ctx)
        response, error = await self._call_provider(ctx)
        return self._finish_call(ctx, response, error)

    async def handle_stream(
        self, payload: dict, session_id: str | None = None
    ) -> AsyncIterator[str]:
        """Streaming request. Yields SSE lines; logs after the stream completes."""
        ctx = self._preflight(payload, session_id)
        if ctx["decision"].action == "A1":
            resp = self._finish_cached(ctx)
            for line in _synthesize_sse(resp):
                yield line
            return

        candidate: Candidate = ctx["candidate"]
        chunks_text: list[str] = []
        usage: dict | None = None
        error: str | None = None
        try:
            import litellm

            stream = await litellm.acompletion(
                model=ctx["model"],
                messages=candidate.messages,
                stream=True,
                stream_options={"include_usage": True},
                **ctx["extra"],
            )
            async for chunk in stream:
                data = _to_dict(chunk)
                if data.get("usage"):
                    usage = data["usage"]
                for choice in data.get("choices", []):
                    delta = (choice.get("delta") or {}).get("content")
                    if delta:
                        chunks_text.append(delta)
                yield f"data: {json.dumps(data)}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as exc:  # degrade: retry raw, non-streaming content as one chunk
            log.warning("stream failed (%s); falling back to raw call", exc)
            error = str(exc)
            ctx["candidate"] = self._raw_candidate(ctx)
            response, err2 = await self._call_provider(ctx)
            error = err2 or error
            text = _response_text(response) if response else ""
            fallback = _chat_response(ctx["model"], text)
            for line in _synthesize_sse(fallback):
                yield line
            chunks_text = [text]
            usage = (response or {}).get("usage")

        full_text = "".join(chunks_text)
        response_dict = _chat_response(ctx["model"], full_text)
        if usage:
            response_dict["usage"] = usage
        self._finish_call(ctx, response_dict, error, streamed=True)

    # ------------------------------------------------------------ pipeline

    def _preflight(self, payload: dict, session_id: str | None) -> dict:
        model = payload.get("model", "gpt-4o")
        messages = payload.get("messages") or []
        provider = provider_of(model)
        session = session_id or _derive_session(messages)
        extra = {k: payload[k] for k in _PASSTHROUGH_KEYS if k in payload}
        query_text = tokens.message_text(messages[-1]) if messages else ""

        x = Features(model=model, provider=provider, session_id=session)
        ledger_pred = LedgerPrediction(0, 0, 0)
        answer_match = context_match = None
        grounding_hits: list = []
        try:
            x = base_features(messages, model, provider, session)
            rule = self.settings.cache_rule_for(provider)
            ledger_pred = self.ledger.predict(session, messages, model, rule)
            x.warm_prefix_tokens = ledger_pred.warm_tokens
            x.tail_tokens = max(x.total_tokens - ledger_pred.warm_tokens, 0)

            exact = self.memory.lookup_exact(model, messages, self.settings.semantic_ttl_s)
            if exact is not None:
                answer_match, x.max_similarity, x.conv_hash_match = exact, 1.0, True
            elif self.embedder is not None and query_text:
                near = self.memory.lookup_semantic(query_text, self.settings.semantic_ttl_s)
                if near is not None:
                    if near.similarity >= self.settings.theta_high:
                        answer_match = near
                        x.max_similarity = near.similarity
                        x.conv_hash_match = near.conv_hash == conversation_hash(messages)
                    elif near.similarity >= self.settings.theta_low:
                        context_match = near
                        x.max_similarity = near.similarity
                        x.context_similarity = near.similarity

            if self.settings.enable_grounding and query_text:
                grounding_hits = self.grounding.query(query_text)
                if grounding_hits:
                    x.grounding_score = grounding_hits[0].score
        except Exception as exc:
            log.warning("analysis failed (%s); proceeding with raw features", exc)

        # Decide.
        try:
            feasible = feasible_actions(
                x, self.settings, context_match is not None, bool(grounding_hits)
            )
            candidates: dict[str, Candidate] = {}
            estimates = {}
            for action in feasible:
                if action == "A1":
                    estimates["A1"] = self.cost_model.estimate(
                        "A1", x, CandidateStats(0, 0)
                    )
                    continue
                cand = self.assembler.build(
                    action, messages, x, ledger_pred, context_match, grounding_hits
                )
                candidates[action] = cand
                estimates[action] = self.cost_model.estimate(action, x, cand.stats)
            decision = choose(estimates, x, self.settings, self.rng)
        except Exception as exc:
            log.warning("decision failed (%s); falling back to A5", exc)
            cand = Candidate(
                "A5",
                messages,
                CandidateStats(ledger_pred.warm_tokens, ledger_pred.cold_tokens),
            )
            candidates = {"A5": cand}
            from preflight.policy.engine import Decision

            decision = Decision("A5", False, {}, ["A5"])

        chosen_candidate = candidates.get(decision.action)
        if chosen_candidate is None and decision.action != "A1":
            chosen_candidate = candidates.get("A5") or Candidate(
                "A5", messages, CandidateStats(0, x.total_tokens)
            )

        return {
            "model": model,
            "provider": provider,
            "session": session,
            "messages": messages,
            "query_text": query_text,
            "extra": extra,
            "features": x,
            "ledger_pred": ledger_pred,
            "decision": decision,
            "candidate": chosen_candidate,
            "answer_match": answer_match,
            "grounding_hits": grounding_hits,
            "t0": time.perf_counter(),
            "payload": payload,
        }

    def _raw_candidate(self, ctx: dict) -> Candidate:
        pred: LedgerPrediction = ctx["ledger_pred"]
        return Candidate(
            "A5", ctx["messages"], CandidateStats(pred.warm_tokens, pred.cold_tokens)
        )

    async def _call_provider(self, ctx: dict) -> tuple[dict | None, str | None]:
        import litellm

        candidate: Candidate = ctx["candidate"]
        try:
            resp = await litellm.acompletion(
                model=ctx["model"], messages=candidate.messages, **ctx["extra"]
            )
            return _to_dict(resp), None
        except Exception as exc:
            if candidate.action != "A5":  # optimized call failed: retry raw once
                log.warning("action %s call failed (%s); retrying raw", candidate.action, exc)
                ctx["candidate"] = self._raw_candidate(ctx)
                try:
                    resp = await litellm.acompletion(
                        model=ctx["model"], messages=ctx["messages"], **ctx["extra"]
                    )
                    return _to_dict(resp), f"degraded from {candidate.action}: {exc}"
                except Exception as exc2:
                    return None, str(exc2)
            return None, str(exc)

    # ------------------------------------------------------------ finishers

    def _finish_cached(self, ctx: dict) -> dict:
        match = ctx["answer_match"]
        response = _chat_response(ctx["model"], match.answer_text)
        response["preflight"] = {"action": "A1", "similarity": round(match.similarity, 4)}
        self._log_outcome(ctx, response_text=match.answer_text, usage=None, error=None)
        self._maybe_audit(ctx)
        return response

    def _finish_call(
        self, ctx: dict, response: dict | None, error: str | None, streamed: bool = False
    ) -> dict:
        if response is None:
            self._log_outcome(ctx, response_text="", usage=None, error=error)
            return {
                "error": {"message": error or "provider call failed", "type": "preflight_upstream"}
            }
        text = _response_text(response)
        usage = response.get("usage")
        self._log_outcome(ctx, response_text=text, usage=usage, error=error)
        self._remember(ctx, text)
        if not streamed and isinstance(response, dict):
            response.setdefault("preflight", {})["action"] = ctx["candidate"].action
        return response

    def _log_outcome(
        self, ctx: dict, response_text: str, usage: dict | None, error: str | None
    ) -> None:
        x: Features = ctx["features"]
        decision = ctx["decision"]
        action = decision.action
        candidate: Candidate | None = ctx.get("candidate")

        if action == "A1" or candidate is None:
            in_miss = in_hit = 0
            out_tokens = 0
        else:
            prompt_tokens = cached_tokens = None
            if usage:
                prompt_tokens = usage.get("prompt_tokens")
                details = usage.get("prompt_tokens_details") or {}
                cached_tokens = details.get("cached_tokens")
            if prompt_tokens is None:
                prompt_tokens = candidate.stats.total
            if cached_tokens is None:
                cached_tokens = min(candidate.stats.warm_tokens, prompt_tokens)
            in_hit = int(cached_tokens)
            in_miss = int(prompt_tokens) - in_hit
            out_tokens = int(
                (usage or {}).get("completion_tokens")
                or tokens.count_text(response_text, x.model)
            )

        realized = self.cost_model.realized_cost(
            x.model, x.provider, in_miss, in_hit, out_tokens
        )
        baseline_stats = CandidateStats(
            ctx["ledger_pred"].warm_tokens, ctx["ledger_pred"].cold_tokens
        )
        baseline = self.cost_model.raw_call_cost(x, baseline_stats)
        estimate = decision.estimates.get(action)

        outcome = Outcome(
            session_id=ctx["session"],
            model=x.model,
            provider=x.provider,
            action=action,
            explored=decision.explored,
            tokens_in_miss=in_miss,
            tokens_in_hit=in_hit,
            tokens_out=out_tokens,
            cost_estimated=estimate.expected_cost if estimate else 0.0,
            cost_realized=realized,
            cost_baseline=baseline,
            latency_ms=(time.perf_counter() - ctx["t0"]) * 1000.0,
            error=error,
            features=x.to_dict(),
            payload=ctx["payload"],
            response_text=response_text[:20000],
        )
        request_id = self.logger.log(outcome)

        try:
            if action != "A1" and out_tokens:
                self.outlen.observe(x, action, out_tokens)
            self._detect_retry(ctx, request_id)
        except Exception as exc:
            log.debug("post-log bookkeeping failed: %s", exc)

    def _remember(self, ctx: dict, response_text: str) -> None:
        """Write T1/T2 answer and T3 context after a successful call."""
        if not response_text:
            return
        try:
            grounding_text = "\n".join(h.text for h in ctx.get("grounding_hits") or [])
            candidate: Candidate = ctx["candidate"]
            self.memory.store_answer(
                model=ctx["model"],
                messages=ctx["messages"],
                query_text=ctx["query_text"],
                answer_text=response_text,
                context={"grounding": grounding_text[:8000], "action": candidate.action},
            )
            self.ledger.record_sent(ctx["session"], candidate.messages, ctx["model"])
        except Exception as exc:
            log.warning("memory/ledger write failed: %s", exc)

    def _detect_retry(self, ctx: dict, request_id: str) -> None:
        """A new query very similar to the previous one in the same session flags
        the previous answer as a suspected failure."""
        session, query = ctx["session"], ctx["query_text"]
        prev = self._last_query.get(session)
        if prev and self.embedder is not None and query:
            prev_query, prev_id = prev
            sim = cosine(self.embedder.embed(query), self.embedder.embed(prev_query))
            if sim >= self.settings.retry_similarity:
                # Flag the *previous* request as a suspected failure; the refit
                # command turns these flags into failure labels per action.
                self.logger.flag_retry(prev_id)
        self._last_query[session] = (query, request_id)

    def _maybe_audit(self, ctx: dict) -> None:
        """Shadow-audit a fraction of A1 hits with a real call (never blocks)."""
        if self.rng.random() >= self.settings.audit_rate:
            return

        async def _audit():
            try:
                import litellm

                resp = await litellm.acompletion(
                    model=ctx["model"], messages=ctx["messages"], **ctx["extra"]
                )
                shadow_text = _response_text(_to_dict(resp))
                match = ctx["answer_match"]
                agreement = 0.0
                if self.embedder is not None and shadow_text:
                    agreement = cosine(
                        self.embedder.embed(shadow_text),
                        self.embedder.embed(match.answer_text),
                    )
                self.logger.log_audit(match.entry_id, agreement, 0.0)
            except Exception as exc:
                log.debug("audit failed: %s", exc)

        try:
            asyncio.get_running_loop().create_task(_audit())
        except RuntimeError:
            pass  # no loop (sync library mode): skip audit


# ---------------------------------------------------------------- helpers


def _derive_session(messages: list[dict]) -> str:
    """Stable session id from the system prompt + first user message."""
    head = [m for m in messages if m.get("role") == "system"][:1]
    first_user = next((m for m in messages if m.get("role") == "user"), None)
    basis = json.dumps([head, first_user], sort_keys=True, default=str)
    return hashlib.sha256(basis.encode()).hexdigest()[:16]


def _to_dict(obj: Any) -> dict:
    if isinstance(obj, dict):
        return obj
    for attr in ("model_dump", "dict", "to_dict"):
        fn = getattr(obj, attr, None)
        if callable(fn):
            try:
                return fn()
            except TypeError:
                continue
    return json.loads(json.dumps(obj, default=lambda o: getattr(o, "__dict__", str(o))))


def _response_text(response: dict | None) -> str:
    if not response:
        return ""
    choices = response.get("choices") or []
    if not choices:
        return ""
    message = choices[0].get("message") or {}
    return message.get("content") or ""


def _chat_response(model: str, text: str) -> dict:
    return {
        "id": f"chatcmpl-preflight-{uuid.uuid4().hex[:12]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }


def _synthesize_sse(response: dict) -> list[str]:
    """Replay a complete response as OpenAI-style SSE chunks."""
    text = _response_text(response)
    rid = response.get("id", f"chatcmpl-preflight-{uuid.uuid4().hex[:12]}")
    model = response.get("model", "")
    base = {"id": rid, "object": "chat.completion.chunk", "model": model}
    lines = []
    first = {
        **base,
        "choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}, "finish_reason": None}],
    }
    lines.append(f"data: {json.dumps(first)}\n\n")
    step = 512
    for i in range(0, max(len(text), 1), step):
        chunk = {
            **base,
            "choices": [{"index": 0, "delta": {"content": text[i : i + step]}, "finish_reason": None}],
        }
        lines.append(f"data: {json.dumps(chunk)}\n\n")
    final = {**base, "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}
    lines.append(f"data: {json.dumps(final)}\n\n")
    lines.append("data: [DONE]\n\n")
    return lines
