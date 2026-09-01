"""Embedding backends.

`auto` uses sentence-transformers when installed and falls back to a deterministic
character n-gram hashing embedder otherwise. The hashing embedder is crude but
dependency-free, which keeps the base install light and the test suite hermetic;
production semantic caching should install the [memory] extra.
"""

from __future__ import annotations

import hashlib
from typing import Protocol

import numpy as np


class Embedder(Protocol):
    dim: int

    def embed(self, text: str) -> np.ndarray: ...


class HashingEmbedder:
    """L2-normalized char-trigram hashing vector. Deterministic, CPU-trivial."""

    def __init__(self, dim: int = 256):
        self.dim = dim

    def embed(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dim, dtype=np.float32)
        t = " " + text.lower().strip() + " "
        for i in range(max(len(t) - 2, 1)):
            gram = t[i : i + 3]
            h = int.from_bytes(hashlib.blake2b(gram.encode(), digest_size=4).digest(), "little")
            vec[h % self.dim] += 1.0
        norm = float(np.linalg.norm(vec))
        return vec / norm if norm > 0 else vec


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str):
        from sentence_transformers import SentenceTransformer  # lazy heavy import

        self._model = SentenceTransformer(model_name)
        dim_fn = getattr(self._model, "get_embedding_dimension", None) or getattr(
            self._model, "get_sentence_embedding_dimension"
        )
        self.dim = int(dim_fn())

    def embed(self, text: str) -> np.ndarray:
        vec = self._model.encode([text], normalize_embeddings=True)[0]
        return np.asarray(vec, dtype=np.float32)


def build_embedder(kind: str, model_name: str, hashing_dim: int) -> Embedder | None:
    if kind == "off":
        return None
    if kind == "hashing":
        return HashingEmbedder(hashing_dim)
    if kind == "sentence-transformers":
        return SentenceTransformerEmbedder(model_name)
    # auto
    try:
        return SentenceTransformerEmbedder(model_name)
    except Exception:
        return HashingEmbedder(hashing_dim)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denom) if denom > 0 else 0.0
