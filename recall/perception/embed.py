"""Image/text embedding backends (open_clip family: MobileCLIP, CLIP, SigLIP2)."""

from __future__ import annotations

import functools
from dataclasses import dataclass

import numpy as np
import torch
from PIL import Image

DEFAULT_MODEL = ("MobileCLIP2-S2", "dfndr2b")


def device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


@dataclass
class Embedder:
    name: str
    pretrained: str
    model: torch.nn.Module
    preprocess: object
    tokenizer: object
    dev: torch.device
    logit_scale: float

    @torch.no_grad()
    def images(self, imgs: list[np.ndarray] | list[Image.Image], batch: int = 128) -> np.ndarray:
        out = []
        for i in range(0, len(imgs), batch):
            chunk = [im if isinstance(im, Image.Image) else Image.fromarray(im) for im in imgs[i : i + batch]]
            x = torch.stack([self.preprocess(im) for im in chunk])  # type: ignore[operator]
            n = len(chunk)
            if n < batch:  # fixed shapes: MPS recompiles its graph for every new batch size
                x = torch.cat([x, x[:1].expand(batch - n, *x.shape[1:])])
            f = self.model.encode_image(x.to(self.dev, self.dtype))  # type: ignore[operator]
            out.append(torch.nn.functional.normalize(f.float(), dim=-1)[:n].cpu().numpy())
        if not out:
            return np.zeros((0, self.dim), np.float32)
        return np.concatenate(out).astype(np.float32)

    @torch.no_grad()
    def texts(self, texts: list[str], batch: int = 256) -> np.ndarray:
        out = []
        for i in range(0, len(texts), batch):
            t = self.tokenizer(texts[i : i + batch]).to(self.dev)  # type: ignore[operator]
            f = self.model.encode_text(t)  # type: ignore[operator]
            out.append(torch.nn.functional.normalize(f.float(), dim=-1).cpu().numpy())
        return np.concatenate(out).astype(np.float32)

    @property
    def dtype(self) -> torch.dtype:
        return next(self.model.parameters()).dtype

    @functools.cached_property
    def dim(self) -> int:
        return int(self.texts(["a"]).shape[1])


@functools.lru_cache(maxsize=4)
def load_embedder(name: str = DEFAULT_MODEL[0], pretrained: str = DEFAULT_MODEL[1]) -> Embedder:
    import open_clip

    dev = device()
    model, _, preprocess = open_clip.create_model_and_transforms(name, pretrained=pretrained)
    model = model.eval().to(dev)
    if dev.type == "mps":
        model = model.half()  # fp16 is ~50x faster than fp32 on MPS here (DECISIONS 2026-10-24)
    tok = open_clip.get_tokenizer(name)
    ls = float(model.logit_scale.exp().item()) if hasattr(model, "logit_scale") else 100.0
    return Embedder(name, pretrained, model, preprocess, tok, dev, ls)


def zero_shot_probs(img_emb: np.ndarray, txt_emb: np.ndarray, scale: float = 100.0) -> np.ndarray:
    logits = scale * img_emb @ txt_emb.T
    logits -= logits.max(axis=-1, keepdims=True)
    p = np.exp(logits)
    return p / p.sum(axis=-1, keepdims=True)
