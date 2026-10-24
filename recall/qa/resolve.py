"""Map a free-text object mention to an EPIC noun class (synonym table first, then text embedding)."""

from __future__ import annotations

import ast
import functools

import numpy as np

from recall.data.queries import readable
from recall.perception.vocab import noun_table


@functools.lru_cache(maxsize=1)
def _synonyms() -> dict[str, int]:
    nt = noun_table()
    out: dict[str, int] = {}
    for i, k, inst in zip(nt.id, nt.key, nt.instances, strict=True):
        out.setdefault(readable(k), int(i))
        out.setdefault(k.replace(":", " "), int(i))
        for s in ast.literal_eval(inst):
            parts = s.split(":")
            out.setdefault(" ".join(reversed(parts)) if len(parts) > 1 else s, int(i))
            out.setdefault(s.replace(":", " "), int(i))
    return out


def resolve(text: str, embedder=None, text_bank: np.ndarray | None = None) -> tuple[int | None, float]:
    """Return (noun class, similarity). Exact/synonym hit → sim 1.0."""
    t = text.lower().strip()
    syn = _synonyms()
    for cand in (t, t.rstrip("s"), t[:-2] if t.endswith("es") else t):
        if cand in syn:
            return syn[cand], 1.0
    if embedder is None or text_bank is None:
        return None, 0.0
    q = embedder.texts([f"a photo of a {t}."])[0]
    s = text_bank @ q
    j = int(np.argmax(s))
    return j, float(s[j])
