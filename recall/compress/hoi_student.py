"""Hand–object interaction (HOI) student: distil the Shan et al. (2020) Faster R-CNN R101 teacher into a
phone-sized network.

Teacher labels are the EPIC-100 released detector outputs (the same ones the offline pipeline uses).
Per frame and per hand side (L, R) the student predicts:

    present logit · 5 contact-state logits (none/self/other/portable/stationary) · hand box · object box

Output layout ``[B, 2, 14]`` = ``present, s0..s4, hand l,t,r,b, obj l,t,r,b`` (boxes normalised 0..1),
which is what ``android/.../ml/Models.kt::LiteRtHandObject`` decodes. Input ``[B, 3, 144, 256]`` RGB in 0..1.

Backbone: MobileNetV3-Small (ImageNet, timm) keeping its 5×8 spatial map so boxes stay localisable.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn

from recall.data.loaders.epic import FrameReader, tar_path
from recall.data.loaders.hoi import PORTABLE, STATIONARY, load_hoi
from recall.paths import CACHE, EPIC_RAW, REPO
from recall.perception.extract import hands_array

IN_HW = (144, 256)
OUT_DIM = 14
DATA = CACHE / "hoi_student"


class HOIStudent(nn.Module):
    def __init__(self, pretrained: bool = True):
        super().__init__()
        import timm

        self.backbone = timm.create_model(
            "mobilenetv3_small_100.lamb_in1k", pretrained=pretrained, features_only=True, out_indices=(4,)
        )
        c = self.backbone.feature_info.channels()[-1]  # type: ignore[union-attr,operator]
        self.reduce = nn.Sequential(nn.Conv2d(c, 32, 1), nn.BatchNorm2d(32), nn.Hardswish())
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(32 * 5 * 8, 256), nn.Hardswish(), nn.Linear(256, 2 * OUT_DIM))
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f = self.backbone((x - self.mean) / self.std)[-1]  # type: ignore[operator]
        y = self.head(self.reduce(f)).view(-1, 2, OUT_DIM)
        boxes = torch.sigmoid(y[..., 6:])
        return torch.cat([y[..., :6], boxes], -1)


def n_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())


# ---------------------------------------------------------------- data


def teacher_targets(fr) -> np.ndarray:
    """[2, 11]: present, state, hand box(4), obj present, obj box(4)."""
    H, O = hands_array(fr)
    out = np.zeros((2, 11), np.float32)
    out[:, 0] = H[:, 0]
    out[:, 1] = H[:, 2]
    out[:, 2:6] = H[:, 3:7]
    out[:, 6:11] = O
    return out


def build_split(split: str, rate: float, max_videos: int | None = None) -> Path:
    """Cache uint8 frames at 144×256 + teacher targets for one split at ``rate`` fps."""
    out = DATA / f"{split}_{rate}.npz"
    if out.exists():
        return out
    DATA.mkdir(parents=True, exist_ok=True)
    subset = [v for v in json.loads((REPO / "configs" / "epic_subset.json").read_text()) if v["split"] == split]
    subset = [v for v in subset if tar_path(v["video_id"]).with_name(v["video_id"] + ".tar.ok").exists()]
    if max_videos:
        subset = subset[:max_videos]
    X, Y, V, T = [], [], [], []
    for v in subset:
        vid, fps = v["video_id"], v["fps"]
        hoi = load_hoi(EPIC_RAW / "hoi" / f"{vid}.pkl")
        rd = FrameReader(vid)
        for t in np.arange(0.0, v["duration"], 1.0 / rate):
            fidx = min(len(rd), int(round(t * fps)) + 1)
            img = Image.fromarray(rd.frame(fidx)).resize(IN_HW[::-1], Image.Resampling.BILINEAR)
            X.append(np.asarray(img))
            Y.append(teacher_targets(hoi[min(fidx - 1, len(hoi) - 1)]))
            V.append(vid)
            T.append(t)
        rd.close()
        print("hoi-data", split, vid, len(X), flush=True)
    np.savez(out, x=np.stack(X), y=np.stack(Y), vid=np.array(V), t=np.array(T, np.float32))
    return out


# ---------------------------------------------------------------- training


def loss_fn(pred: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, dict]:
    present = y[..., 0]
    l_p = nn.functional.binary_cross_entropy_with_logits(pred[..., 0], present)
    m = present > 0
    l_s = nn.functional.cross_entropy(pred[..., 1:6][m], y[..., 1][m].long()) if m.any() else pred.sum() * 0
    l_h = (pred[..., 6:10] - y[..., 2:6]).abs().sum(-1)[m].mean() if m.any() else pred.sum() * 0
    mo = (y[..., 6] > 0) & m
    l_o = (pred[..., 10:14] - y[..., 7:11]).abs().sum(-1)[mo].mean() if mo.any() else pred.sum() * 0
    tot = l_p + l_s + 2 * l_h + 2 * l_o
    return tot, {"present": float(l_p), "state": float(l_s), "hand_box": float(l_h), "obj_box": float(l_o)}


def _aug(x: torch.Tensor) -> torch.Tensor:
    # photometric only (boxes stay valid); no flips — left/right hand identity matters
    b = x.shape[0]
    gain = 1 + 0.25 * (torch.rand(b, 1, 1, 1, device=x.device) - 0.5)
    bias = 0.1 * (torch.rand(b, 1, 1, 1, device=x.device) - 0.5)
    return (x * gain + bias).clamp(0, 1)


def predict(model: nn.Module, x_u8: np.ndarray, dev: torch.device, batch: int = 256) -> np.ndarray:
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(x_u8), batch):
            xb = torch.from_numpy(x_u8[i : i + batch]).to(dev).permute(0, 3, 1, 2).float() / 255
            out.append(model(xb).float().cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, 2, OUT_DIM), np.float32)


def decode(pred: np.ndarray, present_thresh: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
    """Student output [T,2,14] → (hands [T,2,7], hobj [T,2,5]) in the offline-cache layout."""
    T = len(pred)
    H = np.zeros((T, 2, 7), np.float32)
    O = np.zeros((T, 2, 5), np.float32)
    p = 1 / (1 + np.exp(-pred[..., 0]))
    st = pred[..., 1:6].argmax(-1)
    pres = p >= present_thresh
    H[..., 0] = pres
    H[..., 1] = np.where(pres, p, 0)
    H[..., 2] = np.where(pres, st, 0)
    H[..., 3:7] = np.where(pres[..., None], pred[..., 6:10], 0)
    contact = pres & np.isin(st, (PORTABLE, STATIONARY))
    O[..., 0] = contact
    O[..., 1:5] = np.where(contact[..., None], pred[..., 10:14], 0)
    return H, O


def agreement(pred: np.ndarray, y: np.ndarray) -> dict:
    """Student-vs-teacher agreement on held-out frames."""
    H, O = decode(pred)
    tp, sp = y[..., 0] > 0, H[..., 0] > 0
    tc = tp & np.isin(y[..., 1], (PORTABLE, STATIONARY))
    sc = sp & np.isin(H[..., 2], (PORTABLE, STATIONARY))

    def f1(a: np.ndarray, b: np.ndarray) -> float:
        tp_ = float((a & b).sum())
        return 2 * tp_ / max(1.0, float(a.sum() + b.sum()))

    def iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        lt = np.maximum(a[..., :2], b[..., :2])
        rb = np.minimum(a[..., 2:], b[..., 2:])
        inter = np.clip(rb - lt, 0, None).prod(-1)
        ar = lambda z: np.clip(z[..., 2:] - z[..., :2], 0, None).prod(-1)  # noqa: E731
        return inter / np.maximum(ar(a) + ar(b) - inter, 1e-9)

    both = tc & sc & (y[..., 6] > 0)
    ious = iou(O[..., 1:5][both], y[..., 7:11][both]) if both.any() else np.zeros(0)
    frame_contact_t, frame_contact_s = tc.any(1), sc.any(1)
    return {
        "hand_present_f1": f1(tp, sp),
        "contact_f1": f1(tc, sc),
        "frame_contact_f1": f1(frame_contact_t, frame_contact_s),
        "frame_contact_acc": float((frame_contact_t == frame_contact_s).mean()),
        "obj_box_iou_mean": float(ious.mean()) if len(ious) else None,
        "obj_box_iou_ge_0.5": float((ious >= 0.5).mean()) if len(ious) else None,
        "n_frames": len(y),
    }


def train(seed: int = 0, epochs: int = 8, rate_train: float = 0.5, rate_val: float = 0.25) -> tuple[nn.Module, dict]:
    from recall.perception.embed import device

    torch.manual_seed(seed)
    np.random.seed(seed)
    dev = device()
    tr = np.load(build_split("train", rate_train))
    va = np.load(build_split("val", rate_val))
    xtr, ytr = tr["x"], tr["y"]
    model = HOIStudent().to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    bs = 128
    steps = epochs * (len(xtr) // bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=2e-3, total_steps=steps)
    rng = np.random.default_rng(seed)
    log: dict = {"n_train": len(xtr), "n_val": len(va["x"]), "params": n_params(model), "epochs": []}
    t0 = time.time()
    for ep in range(epochs):
        model.train()
        perm = rng.permutation(len(xtr))
        tot = 0.0
        for k in range(len(xtr) // bs):
            idx = np.sort(perm[k * bs : (k + 1) * bs])
            xb = torch.from_numpy(xtr[idx]).to(dev).permute(0, 3, 1, 2).float() / 255
            yb = torch.from_numpy(ytr[idx]).to(dev)
            loss, _ = loss_fn(model(_aug(xb)), yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            tot += float(loss)
        ag = agreement(predict(model, va["x"], dev), va["y"])
        log["epochs"].append({"epoch": ep, "train_loss": tot / max(1, len(xtr) // bs), **ag})
        print("hoi-student", seed, ep, round(tot / max(1, len(xtr) // bs), 4), ag, flush=True)
    log["train_s"] = time.time() - t0
    log["val"] = log["epochs"][-1]
    return model.cpu().eval(), log


def save(model: nn.Module, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)


def load(path: Path) -> HOIStudent:
    m = HOIStudent(pretrained=False)
    m.load_state_dict(torch.load(path, map_location="cpu"))
    return m.eval()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=8)
    a = ap.parse_args()
    model, log = train(a.seed, a.epochs)
    save(model, CACHE / "hoi_student" / f"student_s{a.seed}.pt")
    (CACHE / "hoi_student" / f"student_s{a.seed}.json").write_text(json.dumps(log, indent=1))


if __name__ == "__main__":
    main()
