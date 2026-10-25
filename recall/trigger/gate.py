"""Learned memory trigger μ (Contribution 1): a tiny always-on gate.

Input per 2 fps step t (all cheap, available before any heavy model runs):
  * the 36×64 RGB thumbnail at t and its differences to t−1 and t−2 (9 channels)
  * IMU summary over the last 0.5 s (accel/gyro magnitude mean/std/max + mask) when available

A 4-layer strided CNN maps each step to a 32-d code; a causal temporal conv over
the last 8 codes outputs P(hand–object interaction now). Causal ⇒ runnable
online. ~25 k parameters (≈100 KB fp32, ≈25 KB int8) — far below the 1 MB budget.

Labels: frames inside narrated interaction windows (take / put / open / close /
turn-on / turn-off; EPIC narrations, train participants only), widened by 1 s
before the start so the burst that follows a firing covers the action.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from recall.paths import EPIC_RAW

CTX = 8
IMU_DIM = 7


class TinyGate(nn.Module):
    def __init__(self, width: int = 16, code: int = 32, use_imu: bool = True):
        super().__init__()
        w = width
        self.use_imu = use_imu
        self.cnn = nn.Sequential(
            nn.Conv2d(9, w, 3, 2, 1), nn.BatchNorm2d(w), nn.ReLU(inplace=True),
            nn.Conv2d(w, 2 * w, 3, 2, 1), nn.BatchNorm2d(2 * w), nn.ReLU(inplace=True),
            nn.Conv2d(2 * w, 2 * w, 3, 2, 1), nn.BatchNorm2d(2 * w), nn.ReLU(inplace=True),
            nn.Conv2d(2 * w, code, 3, 2, 1), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
        )  # fmt: skip
        d = code + (IMU_DIM if use_imu else 0)
        self.temporal = nn.Conv1d(d, 32, CTX)  # causal: sees exactly the last CTX steps
        self.head = nn.Sequential(nn.ReLU(), nn.Linear(32, 1))

    def frame_codes(self, x: torch.Tensor, imu: torch.Tensor | None) -> torch.Tensor:
        """x: [N,9,36,64] → [N,D]"""
        c = self.cnn(x)
        if self.use_imu:
            assert imu is not None
            c = torch.cat([c, imu], 1)
        return c

    def forward(self, x: torch.Tensor, imu: torch.Tensor | None) -> torch.Tensor:
        """x: [B,CTX,9,36,64], imu: [B,CTX,IMU_DIM] → logits [B]"""
        B, T = x.shape[:2]
        c = self.frame_codes(x.flatten(0, 1), None if imu is None else imu.flatten(0, 1))
        c = c.view(B, T, -1).transpose(1, 2)
        return self.head(self.temporal(c).squeeze(-1)).squeeze(-1)


def n_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())


def frame_inputs(thumbs: np.ndarray) -> np.ndarray:
    """[T,36,64,3] uint8 → [T,9,36,64] float16: frame, diff(t-1), diff(t-2)."""
    x = thumbs.astype(np.float32) / 255.0
    p1 = np.concatenate([x[:1], x[:-1]])
    p2 = np.concatenate([x[:2], x[:-2]]) if len(x) > 1 else x
    out = np.concatenate([x - 0.5, (x - p1) * 4, (x - p2) * 4], axis=-1)
    return out.transpose(0, 3, 1, 2).astype(np.float16)


def imu_features(video_id: str, t: np.ndarray, win_s: float = 0.5) -> np.ndarray:
    """[T, IMU_DIM]: accel mag mean/std/max, gyro mag mean/std/max, available flag."""
    out = np.zeros((len(t), IMU_DIM), np.float32)
    pa, pg = EPIC_RAW / "imu" / f"{video_id}-accl.csv", EPIC_RAW / "imu" / f"{video_id}-gyro.csv"
    if not (pa.exists() and pg.exists()):
        return out
    for k, p in enumerate((pa, pg)):
        d = pd.read_csv(p)
        ts = d.iloc[:, 0].to_numpy() / 1000.0
        mag = np.linalg.norm(d.iloc[:, 1:4].to_numpy(), axis=1)
        if k == 0:
            mag = mag - 9.81
        lo = np.searchsorted(ts, t - win_s)
        hi = np.searchsorted(ts, t)
        for i, (a, b) in enumerate(zip(lo, hi, strict=True)):
            if b > a:
                seg = mag[a:b]
                out[i, 3 * k : 3 * k + 3] = [seg.mean(), seg.std(), np.abs(seg).max()]
    out[:, :6] = np.tanh(out[:, :6] / np.array([2, 2, 5, 1, 1, 2], np.float32))
    out[:, 6] = 1.0
    return out


def labels(t: np.ndarray, windows: np.ndarray, pre_s: float = 1.0) -> np.ndarray:
    y = np.zeros(len(t), np.float32)
    for a, b in windows:
        y[(t >= a - pre_s) & (t <= b)] = 1.0
    return y


@dataclass
class GateData:
    x: np.ndarray  # [T,9,36,64] fp16
    imu: np.ndarray  # [T,IMU_DIM]
    y: np.ndarray  # [T]
    vid: str


def windows_index(data: list[GateData]) -> list[tuple[int, int]]:
    """All (video, t) with t ≥ 0; context is left-padded by repeating frame 0."""
    return [(k, i) for k, d in enumerate(data) for i in range(len(d.y))]


def batch(data: list[GateData], idx: list[tuple[int, int]]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    X, I, Y = [], [], []
    for k, i in idx:
        d = data[k]
        js = np.clip(np.arange(i - CTX + 1, i + 1), 0, None)
        X.append(d.x[js])
        I.append(d.imu[js])
        Y.append(d.y[i])
    return (torch.from_numpy(np.stack(X).astype(np.float32)), torch.from_numpy(np.stack(I)),
            torch.from_numpy(np.array(Y, np.float32)))  # fmt: skip


def _seq_logits(m: TinyGate, x: torch.Tensor, imu: torch.Tensor | None) -> torch.Tensor:
    """x: [B,L,9,36,64] → causal logits for the last L-CTX+1 steps (identical math to score_video)."""
    B, L = x.shape[:2]
    c = m.frame_codes(x.flatten(0, 1), None if imu is None else imu.flatten(0, 1)).view(B, L, -1).transpose(1, 2)
    return m.head(m.temporal(c).transpose(1, 2)).squeeze(-1)  # [B, L-CTX+1]


def train_gate(
    train: list[GateData],
    val: list[GateData],
    seed: int,
    epochs: int = 6,
    use_imu: bool = True,
    width: int = 16,
    device: str | None = None,
    win: int = 64,
    bs: int = 8,
    log=print,
) -> tuple[TinyGate, dict]:
    """Sequence training: each sample is a contiguous window of ``win`` labelled steps (+CTX-1 left context), so the
    CNN runs once per frame instead of CTX times. One epoch ≈ every training step seen once in expectation."""
    if device is None:
        device = "mps" if torch.backends.mps.is_available() else "cpu"
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    m = TinyGate(width, use_imu=use_imu).to(device)
    pos = float(np.mean(np.concatenate([d.y for d in train])))
    lossf = nn.BCEWithLogitsLoss(pos_weight=torch.tensor((1 - pos) / max(pos, 1e-3), device=device))
    opt = torch.optim.AdamW(m.parameters(), lr=3e-3, weight_decay=1e-4)
    n_steps = sum(len(d.y) for d in train)
    per_epoch = max(1, n_steps // (win * bs))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=3e-3, total_steps=epochs * per_epoch)
    lens = np.array([len(d.y) for d in train], float)
    best, best_state, hist = -1.0, None, []
    for ep in range(epochs):
        m.train()
        tot = 0.0
        for _ in range(per_epoch):
            X, I, Y = [], [], []
            for k in rng.choice(len(train), size=bs, p=lens / lens.sum()):
                d = train[k]
                end = int(rng.integers(1, len(d.y) + 1))
                js = np.clip(np.arange(end - win - CTX + 1, end), 0, None)  # left-pad by repeating frame 0
                X.append(d.x[js])
                I.append(d.imu[js])
                Y.append(d.y[js[CTX - 1 :]])
            x = torch.from_numpy(np.stack(X).astype(np.float32)).to(device)
            im = torch.from_numpy(np.stack(I)).to(device) if use_imu else None
            y = torch.from_numpy(np.stack(Y).astype(np.float32)).to(device)
            loss = lossf(_seq_logits(m, x, im), y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            tot += float(loss.detach())
        m_cpu = m.to("cpu")
        ap = average_precision(m_cpu, val, use_imu, "cpu")
        m = m_cpu.to(device)
        hist.append({"epoch": ep, "train_loss": tot / per_epoch, "val_ap": ap})
        log(f"  seed {seed} imu {use_imu} epoch {ep} loss {hist[-1]['train_loss']:.4f} val AP {ap:.4f}")
        if ap > best:
            best = ap
            best_state = {k: v.detach().cpu().clone() for k, v in m.state_dict().items()}
    assert best_state is not None
    m = m.to("cpu")
    m.load_state_dict(best_state)
    return m.eval(), {"val_ap": best, "history": hist, "pos_rate": pos, "params": n_params(m), "device": device}


@torch.no_grad()
def score_video(m: TinyGate, d: GateData, use_imu: bool, device: str = "cpu", chunk: int = 2048) -> np.ndarray:
    """Causal per-frame scores for a whole video (computes frame codes once, then the temporal conv)."""
    m.eval()
    codes = []
    for i in range(0, len(d.y), chunk):
        x = torch.from_numpy(d.x[i : i + chunk].astype(np.float32)).to(device)
        im = torch.from_numpy(d.imu[i : i + chunk]).to(device) if use_imu else None
        codes.append(m.frame_codes(x, im))
    c = torch.cat(codes).T[None]  # [1,D,T]
    c = torch.cat([c[..., :1].expand(-1, -1, CTX - 1), c], -1)  # left-pad by repeating frame 0
    logits = m.head(m.temporal(c).transpose(1, 2)).squeeze(-1)[0]
    return torch.sigmoid(logits).cpu().numpy()


def average_precision(m: TinyGate, data: list[GateData], use_imu: bool, device: str) -> float:
    from sklearn.metrics import average_precision_score

    s = np.concatenate([score_video(m, d, use_imu, device) for d in data])
    y = np.concatenate([d.y for d in data])
    return float(average_precision_score(y, s))


def save_gate(m: TinyGate, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(m.state_dict(), path)


def skin_score(thumbs: np.ndarray) -> np.ndarray:
    """Cheap hand-presence heuristic: fraction of skin-coloured pixels (YCbCr box rule) in the lower 2/3."""
    x = thumbs[:, thumbs.shape[1] // 3 :].astype(np.float32)
    r, g, b = x[..., 0], x[..., 1], x[..., 2]
    cb = 128 - 0.168736 * r - 0.331264 * g + 0.5 * b
    cr = 128 + 0.5 * r - 0.418688 * g - 0.081312 * b
    skin = (cb >= 77) & (cb <= 127) & (cr >= 133) & (cr <= 173)
    return skin.mean((1, 2)).astype(np.float32)
