"""Proxy-mode example. Start the gateway first:

    OPENAI_API_KEY=sk-... preflight serve

Then run this script - it is a completely standard OpenAI SDK client; the only
Preflight-specific line is the base_url.
"""

from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8411/v1", api_key="unused")

resp = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Explain the prefix ledger idea in one sentence."}],
)
print(resp.choices[0].message.content)

# Streaming works transparently (cached answers are replayed as SSE chunks).
stream = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Count from 1 to 5."}],
    stream=True,
)
for chunk in stream:
    print(chunk.choices[0].delta.content or "", end="", flush=True)
print()
