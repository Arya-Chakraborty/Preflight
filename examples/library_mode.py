"""Minimal library-mode example: no server, no code changes to your app's flow.

    GEMINI_API_KEY=... python examples/library_mode.py
    PREFLIGHT_MODEL=gpt-4o-mini OPENAI_API_KEY=sk-... python examples/library_mode.py
"""

import os

import preflight

MODEL = os.environ.get("PREFLIGHT_MODEL", "gemini/gemini-3.5-flash-lite")

client = preflight.wrap()

for question in [
    "What is prompt caching and why does it matter?",
    "What is prompt caching and why does it matter?",  # exact repeat: served from cache
]:
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": question}],
    )
    if "error" in resp and "choices" not in resp:
        raise SystemExit(f"Provider call failed: {resp['error']['message']}")
    action = resp.get("preflight", {}).get("action", "?")
    print(f"[{action}] {resp['choices'][0]['message']['content'][:80]}...")

print("\nSpend summary:", client.stats())
client.close()
