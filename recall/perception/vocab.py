"""Text vocabularies (EPIC noun classes, location classes) and prompt-ensembled text embeddings."""

from __future__ import annotations

import functools

import numpy as np
import pandas as pd

from recall.data.queries import LOCATION_KEYS, readable
from recall.paths import CACHE, EPIC_ANN
from recall.perception.embed import Embedder

OBJ_TEMPLATES = ["a photo of a {}.", "a close-up photo of a {} held in a hand.", "a {} in a kitchen."]
LOC_TEMPLATES = ["a photo of something in the {}.", "a photo of the {}.", "an object on the {} in a kitchen."]


@functools.lru_cache(maxsize=1)
def noun_table() -> pd.DataFrame:
    return pd.read_csv(EPIC_ANN / "EPIC_100_noun_classes.csv")


def noun_names() -> list[str]:
    return [readable(k) for k in noun_table().key]


def location_ids() -> list[int]:
    k2i = dict(zip(noun_table().key, noun_table().id, strict=True))
    return [int(k2i[k]) for k in LOCATION_KEYS if k in k2i]


def location_names() -> list[str]:
    k2i = dict(zip(noun_table().key, noun_table().id, strict=True))
    return [readable(k) for k in LOCATION_KEYS if k in k2i]


def text_bank(emb: Embedder, names: list[str], templates: list[str], tag: str) -> np.ndarray:
    """Prompt-ensembled, L2-normalised text embeddings, cached on disk per model."""
    path = CACHE / "text" / f"{emb.name}_{tag}_{len(names)}.npy"
    if path.exists():
        return np.load(path)
    T = []
    for tpl in templates:
        T.append(emb.texts([tpl.format(n) for n in names]))
    t = np.mean(T, axis=0)
    t /= np.linalg.norm(t, axis=-1, keepdims=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, t.astype(np.float32))
    return t.astype(np.float32)


def object_bank(emb: Embedder) -> np.ndarray:
    return text_bank(emb, noun_names(), OBJ_TEMPLATES, "obj")


def location_bank(emb: Embedder) -> np.ndarray:
    return text_bank(emb, location_names(), LOC_TEMPLATES, "loc")
