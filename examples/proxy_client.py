"""Proxy-mode example. Start the gateway first:

    GEMINI_API_KEY=... preflight serve

Then run this script - it is a completely standard OpenAI SDK client; the only
Preflight-specific lines are the base_url and the litellm-style model name.
"""

import os

from openai import OpenAI

MODEL = os.environ.get("PREFLIGHT_MODEL", "gemini/gemini-3.5-flash-lite")

client = OpenAI(base_url="http://127.0.0.1:8411/v1", api_key="unused")

resp = client.chat.completions.create(
    model=MODEL,
    messages=[{"role": "user", "content": "Explain the prefix ledger idea in one sentence."}],
)
print(resp.choices[0].message.content)

# Streaming works transparently (cached answers are replayed as SSE chunks).
stream = client.chat.completions.create(
    model=MODEL,
    messages=[{"role": "user", "content": "Count from 1 to 5."}],
    stream=True,
)
for chunk in stream:
    print(chunk.choices[0].delta.content or "", end="", flush=True)
print()
