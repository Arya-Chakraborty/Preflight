"""Minimal library-mode example: no server, no code changes to your app's flow.

    OPENAI_API_KEY=sk-... python examples/library_mode.py
"""

import preflight

client = preflight.wrap()

for question in [
    "What is prompt caching and why does it matter?",
    "What is prompt caching and why does it matter?",  # exact repeat: served from cache
]:
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": question}],
    )
    action = resp.get("preflight", {}).get("action", "?")
    print(f"[{action}] {resp['choices'][0]['message']['content'][:80]}...")

print("\nSpend summary:", client.stats())
