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
import os
import random
import threading
import time
import uuid
from collections.abc import AsyncIterator
from typing import Any

from preflight import tokens
from preflight.analyzer.embeddings import HashingEmbedder, build_embedder, cosine
from preflight.analyzer.features import Features, base_features
from preflight.assembler.assembler import Assembler, Candidate
from preflight.assembler.cache_hints import apply_cache_hints
from preflight.assembler.grounding import GroundingStore
from preflight.config import Settings
from preflight.costs.estimators import load_estimators, refit_from_log
from preflight.costs.model import CandidateStats, CostModel
from preflight.costs.prices import provider_of
from preflight.ledger.ledger import LedgerPrediction, PrefixLedger
from preflight.lock import DataDirLock
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
        self._dir_lock = DataDirLock(settings.data_dir)
        if not settings.allow_multi_writer:
            self._dir_lock.acquire()
        try:
            self._init_stores()
        except BaseException:
            self._dir_lock.release()
            raise

    def _init_stores(self) -> None:
        self.embedder = build_embedder(
            self.settings.embedder, self.settings.embedding_model, self.settings.hashing_dim
        )
        if self.settings.embedder == "auto" and isinstance(self.embedder, HashingEmbedder):
            log.warning(
                "embedder fell back to hashing; install preflight-llm[memory] "
                "for production semantic cache quality",
                extra={"event": "embedder_hashing_fallback"},
            )
        settings = self.settings
        self.memory = MemoryStore(settings.data_dir, self.embedder)
        self.ledger = PrefixLedger(settings.data_dir)
        self.grounding = GroundingStore(settings)
        self.logger = OutcomeLogger(settings.data_dir)
        outlen, pfail = load_estimators(settings)
        self.outlen, self.pfail = outlen, pfail
        self.cost_model = CostModel(settings, outlen, pfail)
        self.assembler = Assembler(settings)
        self.rng = random.Random()
        # session -> recent (query, request_id, features, action)
        self._recent: dict[str, list[tuple[str, str, Features, str]]] = {}
        self._since_refit = 0
        self._bg_loop: asyncio.AbstractEventLoop | None = None
        self._closed = False
        # Serializes estimator mutation/refit and the retry-window bookkeeping,
        # since request finishers run in worker threads (asyncio.to_thread).
        self._est_lock = threading.Lock()
        self._refitting = False
        self._refit_thread: threading.Thread | None = None
        # In-memory spend accounting so the spend cap is O(1) per request instead
        # of a full-table SUM. Global total is seeded once from the log; per-session
        # spend is seeded lazily. See _over_budget / _record_spend.
        self._spend_lock = threading.Lock()
        self._realized_total = float(self.logger.summary()["realized_usd"])
        self._session_spend: dict[str, float] = {}
        # Background audit tasks/futures, drained on close().
        self._bg_tasks: set = set()

    # ------------------------------------------------------------ entrypoints

    async def handle(self, payload: dict, session_id: str | None = None) -> dict:
        """Non-streaming request. Returns an OpenAI-format response dict."""
        self._capture_loop()
        ctx = await asyncio.to_thread(self._preflight, payload, session_id)
        if ctx.get("reject_budget"):
            return {
                "error": {
                    "message": "spend cap exceeded",
                    "type": "preflight_budget",
                },
                "preflight": {"request_id": ctx.get("request_id")},
            }
        if ctx["decision"].action == "A1":
            return await asyncio.to_thread(self._finish_cached, ctx)
        response, error = await self._call_provider(ctx)
        return await asyncio.to_thread(self._finish_call, ctx, response, error)

    async def handle_stream(
        self, payload: dict, session_id: str | None = None
    ) -> AsyncIterator[str]:
        """Streaming request. Yields SSE lines; logs after the stream completes."""
        self._capture_loop()
        ctx = await asyncio.to_thread(self._preflight, payload, session_id)
        if ctx.get("reject_budget"):
            err = {
                "error": {"message": "spend cap exceeded", "type": "preflight_budget"},
                "preflight": {"request_id": ctx.get("request_id")},
            }
            yield f"data: {json.dumps(err)}\n\n"
            yield "data: [DONE]\n\n"
            return
        if ctx["decision"].action == "A1":
            resp = await asyncio.to_thread(self._finish_cached, ctx)
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
                messages=self._outbound_messages(ctx, candidate.messages),
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
            self._attach_meta(fallback, ctx)
            for line in _synthesize_sse(fallback):
                yield line
            chunks_text = [text]
            usage = (response or {}).get("usage")

        full_text = "".join(chunks_text)
        response_dict = _chat_response(ctx["model"], full_text)
        if usage:
            response_dict["usage"] = usage
        await asyncio.to_thread(self._finish_call, ctx, response_dict, error, True)

    # ------------------------------------------------------------ pipeline

    def _preflight(self, payload: dict, session_id: str | None) -> dict:
        model = payload.get("model", "gpt-4o")
        messages = payload.get("messages") or []
        provider = provider_of(model)
        session = session_id or _derive_session(messages)
        extra = {k: payload[k] for k in _PASSTHROUGH_KEYS if k in payload}
        query_text = tokens.message_text(messages[-1]) if messages else ""
        task_type = str(payload.get("task_type") or "")

        x = Features(model=model, provider=provider, session_id=session, task_type=task_type)
        ledger_pred = LedgerPrediction(0, 0, 0)
        answer_match = context_match = None
        grounding_hits: list = []
        analysis_t0 = time.perf_counter()
        try:
            x = base_features(messages, model, provider, session)
            x.task_type = task_type
            rule = self.settings.cache_rule_for(provider)
            ledger_pred = self.ledger.predict(session, messages, model, rule)
            x.warm_prefix_tokens = ledger_pred.warm_tokens
            x.tail_tokens = max(x.total_tokens - ledger_pred.warm_tokens, 0)

            if not self._analyzer_timed_out(analysis_t0):
                exact = self.memory.lookup_exact(model, messages, self.settings.semantic_ttl_s)
                if exact is not None:
                    answer_match, x.max_similarity, x.conv_hash_match = exact, 1.0, True
                    x.matched_answer_id = exact.entry_id
                elif self.embedder is not None and query_text:
                    near = self.memory.lookup_semantic(query_text, self.settings.semantic_ttl_s)
                    if near is not None:
                        if near.similarity >= self.settings.theta_high:
                            answer_match = near
                            x.max_similarity = near.similarity
                            x.conv_hash_match = near.conv_hash == conversation_hash(messages)
                            x.matched_answer_id = near.entry_id
                        elif near.similarity >= self.settings.theta_low:
                            context_match = near
                            x.max_similarity = near.similarity
                            x.context_similarity = near.similarity
                            x.matched_context_id = near.entry_id

            if (
                not self._analyzer_timed_out(analysis_t0)
                and context_match is None
                and self.embedder is not None
                and query_text
            ):
                t3 = self.memory.lookup_context(query_text, self.settings.context_ttl_s)
                if t3 is not None and t3.similarity >= self.settings.theta_low:
                    if context_match is None or t3.similarity > context_match.similarity:
                        context_match = t3
                        x.context_similarity = t3.similarity
                        x.matched_context_id = t3.entry_id
                        if x.max_similarity == 0:
                            x.max_similarity = t3.similarity

            if (
                not self._analyzer_timed_out(analysis_t0)
                and self.settings.enable_grounding
                and query_text
            ):
                grounding_hits = self.grounding.query(query_text)
                if grounding_hits:
                    x.grounding_score = grounding_hits[0].score
        except Exception as exc:
            log.warning("analysis failed (%s); proceeding with raw features", exc)

        # Decide.
        try:
            if self._analyzer_timed_out(analysis_t0) and answer_match is None:
                raise TimeoutError("analyzer budget exhausted")
            feasible = feasible_actions(
                x, self.settings, context_match is not None, bool(grounding_hits)
            )
            candidates: dict[str, Candidate] = {}
            estimates = {}
            assemble_t0 = time.perf_counter()
            for action in feasible:
                if self._assembler_timed_out(assemble_t0):
                    log.warning("assembler budget exhausted; keeping A5 only")
                    feasible = ["A5"]
                    candidates = {}
                    estimates = {}
                    cand = self.assembler.build("A5", messages, x, ledger_pred)
                    candidates["A5"] = cand
                    estimates["A5"] = self.cost_model.estimate("A5", x, cand.stats)
                    break
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
            decision = choose(
                estimates,
                x,
                self.settings,
                self.rng,
                obs_counts=self.pfail.obs_counts(),
                fail_success={a: self.pfail.fail_success_counts(a) for a in estimates},
            )
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

        reject_budget = False
        if decision.action != "A1" and self._over_budget(session):
            reject_budget = True

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
            "request_id": uuid.uuid4().hex,
            "reject_budget": reject_budget,
        }

    def _analyzer_timed_out(self, t0: float) -> bool:
        budget = self.settings.analyzer_timeout_ms
        return budget > 0 and (time.perf_counter() - t0) * 1000.0 > budget

    def _assembler_timed_out(self, t0: float) -> bool:
        budget = self.settings.assembler_timeout_ms
        return budget > 0 and (time.perf_counter() - t0) * 1000.0 > budget

    def _capture_loop(self) -> None:
        """Remember the serving loop so finishers running in worker threads can
        still schedule background coroutines (e.g. A1 shadow audits)."""
        if self._bg_loop is None:
            try:
                self._bg_loop = asyncio.get_running_loop()
            except RuntimeError:
                pass

    def _over_budget(self, session: str) -> bool:
        cap = self.settings.spend_cap_usd
        session_cap = self.settings.session_spend_cap_usd
        if cap is None and session_cap is None:
            return False
        with self._spend_lock:
            if cap is not None and self._realized_total >= cap:
                return True
            if session_cap is None:
                return False
            if session in self._session_spend:
                return self._session_spend[session] >= session_cap
        # Seed outside the lock so a SQLite SUM does not stall other requests.
        seeded = self.logger.session_spend(session)
        with self._spend_lock:
            spent = self._session_spend.setdefault(session, seeded)
            if cap is not None and self._realized_total >= cap:
                return True
            return spent >= session_cap

    def _record_spend(self, session: str, realized: float) -> None:
        """Fold a freshly-logged cost into the in-memory spend counters."""
        with self._spend_lock:
            self._realized_total += realized
            if session in self._session_spend:
                self._session_spend[session] += realized

    def _raw_candidate(self, ctx: dict) -> Candidate:
        pred: LedgerPrediction = ctx["ledger_pred"]
        return Candidate(
            "A5", ctx["messages"], CandidateStats(pred.warm_tokens, pred.cold_tokens)
        )

    def _outbound_messages(self, ctx: dict, messages: list[dict]) -> list[dict]:
        return apply_cache_hints(messages, ctx["provider"], ctx["ledger_pred"])

    async def _call_provider(self, ctx: dict) -> tuple[dict | None, str | None]:
        import litellm

        candidate: Candidate = ctx["candidate"]
        try:
            resp = await litellm.acompletion(
                model=ctx["model"],
                messages=self._outbound_messages(ctx, candidate.messages),
                **ctx["extra"],
            )
            return _to_dict(resp), None
        except Exception as exc:
            if candidate.action != "A5":  # optimized call failed: retry raw once
                log.warning("action %s call failed (%s); retrying raw", candidate.action, exc)
                ctx["candidate"] = self._raw_candidate(ctx)
                try:
                    resp = await litellm.acompletion(
                        model=ctx["model"],
                        messages=self._outbound_messages(ctx, ctx["messages"]),
                        **ctx["extra"],
                    )
                    return _to_dict(resp), f"degraded from {candidate.action}: {exc}"
                except Exception as exc2:
                    return None, str(exc2)
            return None, str(exc)

    # ------------------------------------------------------------ finishers

    def _finish_cached(self, ctx: dict) -> dict:
        match = ctx["answer_match"]
        response = _chat_response(ctx["model"], match.answer_text)
        self._log_outcome(ctx, response_text=match.answer_text, usage=None, error=None)
        self._attach_meta(response, ctx, similarity=match.similarity)
        self._maybe_audit(ctx)
        return response

    def _finish_call(
        self, ctx: dict, response: dict | None, error: str | None, streamed: bool = False
    ) -> dict:
        if response is None:
            self._log_outcome(ctx, response_text="", usage=None, error=error)
            return {
                "error": {"message": error or "provider call failed", "type": "preflight_upstream"},
                "preflight": {"request_id": ctx.get("request_id")},
            }
        text = _response_text(response)
        usage = response.get("usage")
        self._log_outcome(ctx, response_text=text, usage=usage, error=error)
        self._remember(ctx, text)
        if isinstance(response, dict):
            self._attach_meta(response, ctx)
        return response

    def _attach_meta(self, response: dict, ctx: dict, similarity: float | None = None) -> None:
        meta = response.setdefault("preflight", {})
        action = ctx["decision"].action
        if ctx.get("candidate") is not None and action != "A1":
            action = ctx["candidate"].action
        meta["action"] = action
        meta["request_id"] = ctx.get("request_id")
        meta["session"] = ctx.get("session")
        if similarity is not None:
            meta["similarity"] = round(similarity, 4)

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
        # Baseline = what this exact request would have cost sent raw.
        if action == "A5":
            baseline = realized
        else:
            baseline_out = out_tokens
            if action == "A1":
                baseline_out = tokens.count_text(response_text, x.model)
            baseline = self.cost_model.realized_cost(
                x.model,
                x.provider,
                ctx["ledger_pred"].cold_tokens,
                ctx["ledger_pred"].warm_tokens,
                baseline_out,
            )
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
            request_id=ctx.get("request_id") or uuid.uuid4().hex,
        )
        request_id = self.logger.log(outcome)
        ctx["request_id"] = request_id
        self._record_spend(ctx["session"], realized)
        log.info(
            "request finished",
            extra={
                "event": "request",
                "request_id": request_id,
                "session": ctx["session"],
                "action": action,
                "cost_realized": round(realized, 8),
                "cost_baseline": round(baseline, 8),
                "latency_ms": round(outcome.latency_ms, 2),
                "error": error,
            },
        )

        try:
            with self._est_lock:
                persist = not self._refitting
                if action != "A1" and out_tokens:
                    self.outlen.observe(x, action, out_tokens, persist=persist)
                self.pfail.observe(x, action, failed=bool(error), persist=persist)
                self._detect_retry(ctx, request_id, x, action, persist=persist)
                self._maybe_auto_refit()
        except Exception as exc:
            log.debug("post-log bookkeeping failed: %s", exc)

    def _remember(self, ctx: dict, response_text: str) -> None:
        """Write T1/T2 answer and T3 context after a successful call."""
        if not response_text:
            return
        try:
            payload = _t3_payload(ctx, response_text)
            candidate: Candidate = ctx["candidate"]
            self.memory.store_answer(
                model=ctx["model"],
                messages=ctx["messages"],
                query_text=ctx["query_text"],
                answer_text=response_text,
                context=payload,
            )
            if any(payload.get(k) for k in ("grounding", "tools", "reasoning")):
                self.memory.store_context(
                    model=ctx["model"],
                    messages=ctx["messages"],
                    query_text=ctx["query_text"],
                    context=payload,
                )
            self.ledger.record_sent(ctx["session"], candidate.messages, ctx["model"])
        except Exception as exc:
            log.warning("memory/ledger write failed: %s", exc)

    def _detect_retry(
        self, ctx: dict, request_id: str, x: Features, action: str, persist: bool = True
    ) -> None:
        """A new query similar to one of the last `retry_window` in the session
        flags that earlier answer as a suspected failure."""
        session, query = ctx["session"], ctx["query_text"]
        window = max(int(self.settings.retry_window), 1)
        history = self._recent.get(session, [])
        if self.embedder is not None and query:
            q_emb = self.embedder.embed(query)
            for prev_query, prev_id, prev_x, prev_action in reversed(history[-window:]):
                sim = cosine(q_emb, self.embedder.embed(prev_query))
                if sim >= self.settings.retry_similarity:
                    self.logger.flag_retry(prev_id)
                    self.pfail.revise_to_failed(prev_action, persist=persist)
                    break
        history.append((query, request_id, x, action))
        self._recent[session] = history[-window:]

    def _maybe_auto_refit(self) -> None:
        """Trigger an estimator refit off the hot path. Called while holding
        `_est_lock`; the heavy read/fit runs in a daemon thread so the triggering
        request's finish isn't stalled by a full-log scan."""
        every = self.settings.auto_refit_every
        if every <= 0:
            return
        self._since_refit += 1
        if self._since_refit < every or self._refitting:
            return
        self._since_refit = 0
        self._refitting = True
        self._refit_thread = threading.Thread(target=self._run_refit, daemon=True)
        self._refit_thread.start()

    def _run_refit(self) -> None:
        try:
            refit_from_log(self.logger, self.settings)
            outlen, pfail = load_estimators(self.settings)
            with self._est_lock:
                self.outlen, self.pfail = outlen, pfail
                self.cost_model = CostModel(self.settings, self.outlen, self.pfail)
        except Exception as exc:
            log.debug("auto-refit failed: %s", exc)
        finally:
            self._refitting = False

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
                rid = ctx.get("request_id") or match.entry_id
                self.logger.log_audit(rid, agreement, 0.0)
                if agreement < 0.5:
                    with self._est_lock:
                        self.pfail.revise_to_failed("A1", persist=not self._refitting)
            except Exception as exc:
                log.debug("audit failed: %s", exc)

        self._schedule(_audit())

    def _schedule(self, coro) -> None:
        """Fire-and-forget a coroutine, tracking it so close() can cancel it."""
        if self._closed:
            coro.close()
            return
        try:
            loop = asyncio.get_running_loop()
            handle = loop.create_task(coro)
        except RuntimeError:
            if self._bg_loop is not None and self._bg_loop.is_running():
                handle = asyncio.run_coroutine_threadsafe(coro, self._bg_loop)
            else:
                coro.close()
                return
        self._bg_tasks.add(handle)
        handle.add_done_callback(self._bg_tasks.discard)

    def _cancel_bg_tasks(self) -> None:
        handles = list(self._bg_tasks)
        self._bg_tasks.clear()

        def _cancel() -> None:
            for handle in handles:
                try:
                    handle.cancel()
                except Exception:
                    pass

        loop = self._bg_loop
        try:
            running = asyncio.get_running_loop()
        except RuntimeError:
            running = None
        # Task.cancel() is not thread-safe; hop onto the serving loop when needed.
        if loop is not None and running is not loop and loop.is_running():
            loop.call_soon_threadsafe(_cancel)
        else:
            _cancel()

    def close(self) -> None:
        """Cancel background work and release the data-dir lock. Idempotent."""
        if self._closed:
            return
        self._closed = True
        self._cancel_bg_tasks()
        t = self._refit_thread
        if t is not None and t.is_alive():
            t.join(timeout=5.0)
        self._dir_lock.release()

    def readiness(self) -> dict:
        """Checks used by GET /ready. `ok` is True only if every probe passed."""
        reasons: list[str] = []
        data_dir = self.settings.data_dir
        if not data_dir.is_dir() or not os.access(data_dir, os.W_OK):
            reasons.append("data_dir not writable")
        if not self.settings.allow_multi_writer and not self._dir_lock.held:
            reasons.append("data_dir lock not held")
        for name, fn in (
            ("memory", self.memory.ping),
            ("ledger", self.ledger.ping),
            ("outcomes", self.logger.ping),
            ("grounding", self.grounding.ping),
        ):
            try:
                fn()
            except Exception as exc:
                reasons.append(f"{name} sqlite: {exc}")
        if self.settings.embedder != "off" and self.embedder is None:
            reasons.append("embedder failed to load")
        return {
            "status": "ok" if not reasons else "not_ready",
            "ok": not reasons,
            "reasons": reasons,
            "lock_held": self._dir_lock.held,
            "embedder": type(self.embedder).__name__ if self.embedder else None,
        }


# ---------------------------------------------------------------- helpers


def _t3_payload(ctx: dict, response_text: str) -> dict:
    messages = ctx.get("messages") or []
    grounding_text = "\n".join(h.text for h in ctx.get("grounding_hits") or [])
    tools: list[str] = []
    for msg in messages:
        if msg.get("role") == "tool":
            tools.append(tokens.message_text(msg)[:2000])
        if msg.get("tool_calls"):
            tools.append(json.dumps(msg["tool_calls"], default=str)[:2000])
    reasoning = ""
    for msg in reversed(messages[:-1]):
        if msg.get("role") == "assistant":
            reasoning = tokens.message_text(msg)[:2000]
            break
    candidate = ctx.get("candidate")
    return {
        "grounding": grounding_text[:8000],
        "tools": "\n".join(tools)[:4000],
        "reasoning": reasoning,
        "action": getattr(candidate, "action", ""),
        "answer": response_text[:2000],
    }


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
    if response.get("preflight"):
        base["preflight"] = response["preflight"]
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
