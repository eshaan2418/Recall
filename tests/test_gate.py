import numpy as np
import torch

from recall.trigger.gate import (
    CTX,
    IMU_DIM,
    GateData,
    TinyGate,
    frame_inputs,
    n_params,
    score_video,
    skin_score,
    train_gate,
)


def _data(T=300, seed=0):
    rng = np.random.default_rng(seed)
    th = rng.integers(0, 50, size=(T, 36, 64, 3)).astype(np.uint8)
    y = np.zeros(T, np.float32)
    for s in range(20, T - 20, 40):
        th[s : s + 6] += 120  # bright "interaction" frames
        y[s : s + 6] = 1
    return GateData(frame_inputs(th), np.zeros((T, IMU_DIM), np.float32), y, "S")


def test_gate_small():
    m = TinyGate()
    assert n_params(m) < 250_000  # ≤1 MB fp32
    x = torch.zeros(2, CTX, 9, 36, 64)
    assert m(x, torch.zeros(2, CTX, IMU_DIM)).shape == (2,)


def test_score_video_matches_forward_and_is_causal():
    torch.manual_seed(0)
    m = TinyGate().eval()
    d = _data(60)
    s = score_video(m, d, True)
    i = 30
    js = np.clip(np.arange(i - CTX + 1, i + 1), 0, None)
    x = torch.from_numpy(d.x[js].astype(np.float32))[None]
    p = torch.sigmoid(m(x, torch.from_numpy(d.imu[js])[None])).item()
    assert abs(p - s[i]) < 1e-4
    d2 = GateData(d.x.copy(), d.imu, d.y, "S")
    d2.x[i + 1 :] = 0  # changing the future must not change score at i
    assert abs(score_video(m, d2, True)[i] - s[i]) < 1e-5


def test_gate_learns_synthetic():
    m, log = train_gate([_data(400, 0)], [_data(200, 1)], seed=0, epochs=3, bs=8, win=32, device="cpu", log=lambda *_: None)
    assert log["val_ap"] > 0.6


def test_skin_score_range():
    th = np.zeros((3, 36, 64, 3), np.uint8)
    th[1] = (200, 150, 120)
    s = skin_score(th)
    assert s[0] == 0 and s[1] > 0.9
