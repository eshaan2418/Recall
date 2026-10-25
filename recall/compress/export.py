"""Export Recall's on-device models to LiteRT (.tflite) at several precisions.

Run with the export environment (``.venv-export``: litert-torch + ai-edge-quantizer + ai-edge-litert):

    PYTHONPATH=. .venv-export/bin/python -m recall.compress.export --all

Models and I/O contracts (must match ``android/app/src/main/java/ai/recall/ml/Models.kt``):

* ``embedder``    MobileCLIP2-S2 image tower. in ``[1,3,256,256]`` RGB 0..1 (open_clip eval transform:
                  resize short side → 256, centre crop; mean 0 / std 1) → out ``[1,512]`` L2-normalised.
* ``gate``        TinyGate. in ``x [1,8,9,36,64]``, ``imu [1,8,7]`` → out logit ``[1]``.
* ``hoi_student`` HOIStudent. in ``[1,3,144,256]`` RGB 0..1 → out ``[1,2,14]``.

Precisions: ``fp32`` · ``int8dyn`` (int8 weights, dynamic-range activations) · ``int8`` (static full-int8
weights+activations, calibrated on train-split inputs only) · ``int4w`` (int4 channel-wise weights, fp32 act).
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch

from recall.paths import CACHE, REPO

OUT = REPO / "models" / "litert"
PRECISIONS = ("fp32", "int8dyn", "int8", "int4w")


class _Embed(torch.nn.Module):
    def __init__(self, visual: torch.nn.Module):
        super().__init__()
        self.visual = visual

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.normalize(self.visual(x), dim=-1)


def embedder_module() -> torch.nn.Module:
    import open_clip

    m, _, _ = open_clip.create_model_and_transforms("MobileCLIP2-S2", pretrained="dfndr2b")
    return _Embed(m.visual.float().eval()).eval()


def gate_module(path: Path) -> torch.nn.Module:
    from recall.trigger.gate import TinyGate

    g = TinyGate(use_imu=True)
    g.load_state_dict(torch.load(path, map_location="cpu"))
    return g.float().eval()


def hoi_module(path: Path) -> torch.nn.Module:
    from recall.compress.hoi_student import load

    return load(path).float().eval()


def convert_fp32(module: torch.nn.Module, sample: tuple[torch.Tensor, ...], out: Path) -> Path:
    import litert_torch

    out.parent.mkdir(parents=True, exist_ok=True)
    litert_torch.convert(module, sample).export(str(out))
    return out


def quantize(fp32: Path, precision: str, calib: list[tuple[np.ndarray, ...]] | None, out: Path) -> Path:
    from ai_edge_quantizer import quantizer, recipe

    if precision == "int8dyn":
        rcp = recipe.dynamic_wi8_afp32()
    elif precision == "int4w":
        rcp = recipe.dynamic_wi4_afp32()
    elif precision == "int8":
        rcp = recipe.static_wi8_ai8()
    else:
        raise ValueError(precision)
    q = quantizer.Quantizer(str(fp32), rcp)
    res = None
    if q.need_calibration:
        assert calib, "static int8 needs calibration samples"
        sig = _signature_inputs(fp32)
        data = {"serving_default": [dict(zip(sig, c, strict=True)) for c in calib]}
        res = q.calibrate(data)
    out.unlink(missing_ok=True)
    q.quantize(res).export_model(str(out))
    return out


def _signature_inputs(path: Path) -> list[str]:
    from ai_edge_litert.interpreter import Interpreter

    it = Interpreter(model_path=str(path))
    runner = it.get_signature_runner()
    det = runner.get_input_details()
    # signature input names are ordered as args0, args1, ...
    return sorted(det, key=lambda k: (len(k), k))


class TFLiteRunner:
    """Batch-1 LiteRT runner (CPU/XNNPACK), mirroring how the phone invokes the model."""

    def __init__(self, path: Path, threads: int = 4):
        from ai_edge_litert.interpreter import Interpreter

        self.path = path
        self.it = Interpreter(model_path=str(path), num_threads=threads)
        self.runner = self.it.get_signature_runner()
        det = self.runner.get_input_details()
        self.inputs = sorted(det, key=lambda k: (len(k), k))
        self.in_q = {k: (det[k]["dtype"], *det[k]["quantization"]) for k in self.inputs}
        odet = self.runner.get_output_details()
        self.out_key = next(iter(odet))
        self.out_q = (odet[self.out_key]["dtype"], *odet[self.out_key]["quantization"])

    def __call__(self, *xs: np.ndarray) -> np.ndarray:
        feed = {}
        for k, x in zip(self.inputs, xs, strict=True):
            dt, scale, zp = self.in_q[k]
            if np.issubdtype(dt, np.integer):  # full-int8 model: quantise at the boundary, as the app would
                info = np.iinfo(dt)
                x = np.clip(np.round(np.asarray(x, np.float32) / scale + zp), info.min, info.max)
            feed[k] = np.ascontiguousarray(x, dt)
        y = np.asarray(self.runner(**feed)[self.out_key])
        dt, scale, zp = self.out_q
        if np.issubdtype(dt, np.integer):
            y = (y.astype(np.float32) - zp) * scale
        return y


def tflite_image_fn(path: Path, threads: int = 4) -> Callable[..., np.ndarray]:
    """``images_fn`` for :func:`recall.perception.extract.extract_with` backed by a LiteRT embedder."""
    import open_clip
    from PIL import Image

    _, _, pp = open_clip.create_model_and_transforms("MobileCLIP2-S2", pretrained=None)
    run = TFLiteRunner(path, threads)

    def images(imgs: list, batch: int = 1) -> np.ndarray:
        out = []
        for im in imgs:
            x = pp(im if isinstance(im, Image.Image) else Image.fromarray(im)).numpy()[None]  # type: ignore[operator]
            out.append(run(x)[0])
        return np.stack(out).astype(np.float32) if out else np.zeros((0, 512), np.float32)

    return images


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def calib_embedder(n: int = 64) -> list[tuple[np.ndarray, ...]]:
    """Representative crops for static quantisation: train-split frames only."""
    import open_clip
    from PIL import Image

    from recall.bench.harness import subset
    from recall.data.loaders.epic import FrameReader

    _, _, pp = open_clip.create_model_and_transforms("MobileCLIP2-S2", pretrained=None)
    rng = np.random.default_rng(0)
    vids = [v for v in subset("train") if (CACHE / "feats" / "MobileCLIP2-S2" / f"{v['video_id']}.npz").exists()][:4]
    out: list[tuple[np.ndarray, ...]] = []
    for v in vids:
        rd = FrameReader(v["video_id"])
        for f in rng.integers(1, len(rd), n // len(vids)):
            img = rd.frame(int(f))
            if rng.random() < 0.5:  # mix full frames and object-sized crops (what the phone embeds)
                h, w = img.shape[:2]
                cw, ch = int(w * rng.uniform(0.15, 0.4)), int(h * rng.uniform(0.2, 0.5))
                x0, y0 = int(rng.integers(0, w - cw)), int(rng.integers(0, h - ch))
                img = img[y0 : y0 + ch, x0 : x0 + cw]
            out.append((pp(Image.fromarray(img)).numpy()[None],))  # type: ignore[operator]
        rd.close()
    return out


def calib_gate(n: int = 64) -> list[tuple[np.ndarray, ...]]:
    from recall.bench.harness import subset
    from recall.trigger.gate import CTX, frame_inputs, imu_features

    rng = np.random.default_rng(0)
    out: list[tuple[np.ndarray, ...]] = []
    vids = [v for v in subset("train") if (CACHE / "feats" / "MobileCLIP2-S2" / f"{v['video_id']}.npz").exists()][:4]
    for v in vids:
        z = np.load(CACHE / "feats" / "MobileCLIP2-S2" / f"{v['video_id']}.npz")
        x = frame_inputs(z["thumb"]).astype(np.float32)
        imu = imu_features(v["video_id"], z["t"])
        for i in rng.integers(CTX, len(x), n // len(vids)):
            out.append((x[i - CTX + 1 : i + 1][None], imu[i - CTX + 1 : i + 1][None]))
    return out


def calib_hoi(n: int = 64) -> list[tuple[np.ndarray, ...]]:
    from recall.compress.hoi_student import DATA

    z = np.load(sorted(DATA.glob("train_*.npz"))[0], mmap_mode="r")
    idx = np.random.default_rng(0).choice(len(z["x"]), n, replace=False)
    return [((z["x"][i].transpose(2, 0, 1)[None] / 255.0).astype(np.float32),) for i in np.sort(idx)]


def export_all(names: list[str], gate_pt: Path, hoi_pt: Path) -> dict:
    mf = OUT / "MANIFEST.json"
    manifest: dict = json.loads(mf.read_text()) if mf.exists() else {}
    specs = {
        "embedder": (embedder_module, (torch.rand(1, 3, 256, 256),), calib_embedder),
        "gate": (lambda: gate_module(gate_pt), (torch.rand(1, 8, 9, 36, 64) - 0.5, torch.rand(1, 8, 7)), calib_gate),
        "hoi_student": (lambda: hoi_module(hoi_pt), (torch.rand(1, 3, 144, 256),), calib_hoi),
    }
    for name in names:
        build, sample, calib_fn = specs[name]
        mod = build()
        fp32 = convert_fp32(mod, sample, OUT / f"{name}_fp32.tflite")
        calib = calib_fn()
        files = {"fp32": fp32}
        for prec in PRECISIONS[1:]:
            try:
                files[prec] = quantize(fp32, prec, calib if prec == "int8" else None, OUT / f"{name}_{prec}.tflite")
            except Exception as e:  # record, don't hide: the cost table shows which precisions failed
                manifest.setdefault("failures", {})[f"{name}_{prec}"] = repr(e)[:400]
                print("quantize failed", name, prec, e)
        # numerical parity vs PyTorch on the calibration-distribution inputs
        ref = [mod(*[torch.from_numpy(np.asarray(a)) for a in c]).detach().numpy() for c in calib[:16]]
        for prec, p in files.items():
            run = TFLiteRunner(p)
            got = [run(*c) for c in calib[:16]]
            err = float(np.mean([np.abs(g - r).max() for g, r in zip(got, ref, strict=True)]))
            cos = None
            if name == "embedder":
                cos = float(np.mean([float(g.ravel() @ r.ravel()) for g, r in zip(got, ref, strict=True)]))
            manifest[f"{name}_{prec}"] = {
                "file": p.name,
                "bytes": p.stat().st_size,
                "sha256": sha256(p),
                "inputs": [list(s.shape) for s in sample],
                "max_abs_err_vs_torch": err,
                "cosine_vs_torch": cos,
            }
            print(name, prec, manifest[f"{name}_{prec}"])
    mf.write_text(json.dumps(manifest, indent=1))
    return manifest


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=["embedder", "gate", "hoi_student"])
    ap.add_argument("--gate-pt", default=str(CACHE / "gates" / "learned+imu_s0.pt"))
    ap.add_argument("--hoi-pt", default=str(CACHE / "hoi_student" / "student_s0.pt"))
    a = ap.parse_args()
    export_all(a.models, Path(a.gate_pt), Path(a.hoi_pt))


if __name__ == "__main__":
    main()
