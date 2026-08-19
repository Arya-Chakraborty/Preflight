"""Compression backends with a structured-content bypass.

Token-pruning compressors (LLMLingua) corrupt JSON and code, so structured text
is routed through deterministic cleanup only (whitespace collapse, exact-duplicate
line removal) which preserves syntax. Plain prose goes through LLMLingua-2 when
the [compression] extra is installed, deterministic cleanup otherwise.
"""

from __future__ import annotations

import re

from preflight.analyzer.features import structure_fraction

_STRUCTURE_THRESHOLD = 0.5


def deterministic_clean(text: str, dedup: bool = True) -> str:
    """Cleanup that never paraphrases: collapse whitespace runs and (for prose
    only) drop exact-duplicate lines, which are common in tool output.

    Dedup is disabled for structured content: removing a repeated element line
    from a JSON array would corrupt syntax.
    """
    lines = text.splitlines()
    seen: set[str] = set()
    out: list[str] = []
    for line in lines:
        collapsed = re.sub(r"[ \t]{2,}", " ", line.rstrip())
        key = collapsed.strip()
        if dedup and key and key in seen and len(key) > 20:
            continue
        if key:
            seen.add(key)
        out.append(collapsed)
    cleaned = "\n".join(out)
    return re.sub(r"\n{3,}", "\n\n", cleaned)


class Compressor:
    def __init__(self, rate: float = 0.5):
        self._rate = rate
        self._lingua = None
        self._lingua_tried = False

    def _get_lingua(self):
        if not self._lingua_tried:
            self._lingua_tried = True
            try:
                from llmlingua import PromptCompressor  # heavy optional import

                self._lingua = PromptCompressor(
                    model_name="microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank",
                    use_llmlingua2=True,
                )
            except Exception:
                self._lingua = None
        return self._lingua

    def compress(self, text: str) -> str:
        """Compress prose; bypass token pruning for structured content."""
        if not text.strip():
            return text
        if structure_fraction(text) >= _STRUCTURE_THRESHOLD:
            return deterministic_clean(text, dedup=False)
        lingua = self._get_lingua()
        if lingua is not None:
            try:
                result = lingua.compress_prompt(text, rate=self._rate)
                compressed = result.get("compressed_prompt", "")
                if compressed.strip():
                    return compressed
            except Exception:
                pass
        return deterministic_clean(text)
