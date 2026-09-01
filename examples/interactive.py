"""Interactive live test: chat with the gateway and watch the accounting.

    GEMINI_API_KEY=... python examples/interactive.py
    PREFLIGHT_MODEL=gemini/gemini-3.5-flash-lite python examples/interactive.py

After every answer you see: the action taken (A1 cache / A2 context-reuse /
A3 compress / A4 ground / A5 raw), the token split (provider-cache hits vs
misses), realized dollars, what the request would have cost raw, and running
session totals.

Commands:
    /new    start a fresh conversation (new session for the prefix ledger)
    /stats  full summary from the outcome log
    /quit   exit
"""

from __future__ import annotations

import os
import uuid

import preflight

MODEL = os.environ.get("PREFLIGHT_MODEL", "gemini/gemini-3.5-flash-lite")
SYSTEM = "You are a concise, helpful assistant."


def main() -> None:
    client = preflight.wrap()
    logger = client.gateway.logger

    session = uuid.uuid4().hex[:8]
    messages: list[dict] = [{"role": "system", "content": SYSTEM}]
    total_spent = 0.0
    total_raw = 0.0

    print(f"Preflight interactive test | model={MODEL}")
    print("Type a question, or /new, /stats, /quit. Repeat a question to see the cache work.\n")

    while True:
        try:
            query = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not query:
            continue
        if query == "/quit":
            break
        if query == "/new":
            session = uuid.uuid4().hex[:8]
            messages = [{"role": "system", "content": SYSTEM}]
            print("(new conversation)\n")
            continue
        if query == "/stats":
            s = logger.summary()
            print(f"\n  requests={s['requests']}  spend=${s['realized_usd']:.6f}  "
                  f"raw-equivalent=${s['baseline_usd']:.6f}  "
                  f"saved=${s['baseline_usd'] - s['realized_usd']:.6f}")
            for action, row in sorted(s["by_action"].items()):
                print(f"  {action}: n={row['n']}  ${row['usd']:.6f}")
            print()
            continue

        messages.append({"role": "user", "content": query})
        resp = client.chat.completions.create(
            model=MODEL, messages=list(messages), session_id=session
        )
        if "error" in resp and "choices" not in resp:
            print(f"  !! provider error: {resp['error']['message']}\n")
            messages.pop()
            continue

        answer = resp["choices"][0]["message"]["content"]
        messages.append({"role": "assistant", "content": answer})
        print(f"\n{answer}\n")

        row = logger.rows()[-1]
        spent = row["cost_realized"] or 0.0
        raw = row["cost_baseline"] or 0.0
        total_spent += spent
        total_raw += raw
        saved_pct = 100 * (1 - total_spent / total_raw) if total_raw else 0.0
        try:
            import json as _json

            feats = _json.loads(row["features_json"] or "{}")
            sim = feats.get("max_similarity", 0.0)
            conv_ok = feats.get("conv_hash_match", False)
            sim_note = f"sim={sim:.2f}{'' if conv_ok or sim == 0 else ' (context mismatch)'}"
        except Exception:
            sim_note = "sim=?"
        print(
            f"  [{row['action']}] {sim_note} | in={row['tokens_in_miss']} miss / "
            f"{row['tokens_in_hit']} cached | out={row['tokens_out']} tok | "
            f"this=${spent:.6f} (raw ${raw:.6f}) | latency={row['latency_ms']:.0f} ms"
        )
        print(
            f"  session: spent=${total_spent:.6f}  raw-equivalent=${total_raw:.6f}  "
            f"saved={saved_pct:.1f}%\n"
        )


if __name__ == "__main__":
    main()
