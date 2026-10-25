"""M8 (fallback) — analytic energy model. EVERYTHING HERE IS `estimated`, NOT MEASURED.

No phone was available (BLOCKERS.md), so battery %/hour is estimated from

    E_hour = P_capture·3600 + N_gate·E_gate + N_heavy·E_heavy + N_events·E_store

where the *counts* (gate invocations, heavy invocations, events per hour) come from the offline evaluation
(`results/m2/pareto_test.json`), the *work* per invocation (MACs) is counted exactly from the PyTorch models,
and the *energy per unit of work* and *capture power* are ASSUMPTIONS given as low / mid / high ranges:

* capture power (sensor + ISP + frame delivery at low resolution, 2 fps): image sensing costs "hundreds of
  milliwatts" (LiKamWa et al., MobiSys 2013); 2010s phones drew ~1.4–1.65 W for whole-phone camera recording
  (reported in smartphone energy studies; see paper related work). We sweep 0.05 W (glasses-class dedicated
  low-power sensor) / 0.3 W / 1.0 W (phone camera pipeline kept open).
* energy per GMAC: 2 mJ (NPU) / 20 mJ (mobile GPU) / 120 mJ (mobile CPU). These are order-of-magnitude
  assumptions, not vendor numbers.
* battery: 17.3 Wh (4,500 mAh × 3.85 V, a typical current flagship) — assumption.

The on-device harness (``recall/energy/harness.py``) measures the real values when a phone is connected and
replaces this model (`on-device` evidence stage).
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import torch

from recall.bench.common import result_header, write_result
from recall.paths import RESULTS

BATTERY_WH = 17.3
CAPTURE_W = {"low": 0.05, "mid": 0.3, "high": 1.0}
J_PER_GMAC = {"npu": 0.002, "gpu": 0.02, "cpu": 0.12}
E_STORE_J = 0.01  # one SQLite insert + 4 KB JPEG write; small either way (assumption)
REGIONS_PER_HEAVY = 1.0  # measured below from the feature cache when available
STEP_HZ = 2.0


def gmacs(module: torch.nn.Module, *inputs: torch.Tensor) -> float:
    from torch.utils.flop_counter import FlopCounterMode

    with FlopCounterMode(display=False) as fc, torch.no_grad():
        module(*inputs)
    return fc.get_total_flops() / 2 / 1e9


def model_work() -> dict[str, float]:
    import open_clip

    from recall.compress.hoi_student import HOIStudent
    from recall.trigger.gate import TinyGate

    m, _, _ = open_clip.create_model_and_transforms("MobileCLIP2-S2", pretrained=None)
    g = TinyGate(use_imu=True).eval()
    return {
        "embedder_gmac": gmacs(m.visual.eval(), torch.rand(1, 3, 256, 256)),
        "hoi_student_gmac": gmacs(HOIStudent(pretrained=False).eval(), torch.rand(1, 3, 144, 256)),
        # gate per step: CNN on 1 new frame (codes for older frames are cached) + temporal conv
        "gate_gmac_per_step": gmacs(g, torch.rand(1, 8, 9, 36, 64), torch.rand(1, 8, 7)) / 8,
    }


def regions_per_heavy_frame() -> float:
    from recall.bench.harness import subset
    from recall.paths import CACHE

    n_reg = n_fr = 0
    for v in subset("test"):
        p = CACHE / "feats" / "MobileCLIP2-S2" / f"{v['video_id']}.npz"
        if p.exists():
            z = np.load(p)
            n_reg += len(z["reg_t"])
            n_fr += len(z["t"])
    return n_reg / max(1, n_fr)


def energy_per_hour(
    heavy_per_hour: float, events_per_hour: float, work: dict, regions: float, accel: str, capture: str
) -> dict:
    j = J_PER_GMAC[accel]
    e_heavy = (work["hoi_student_gmac"] + (1 + regions) * work["embedder_gmac"]) * j
    e_gate = work["gate_gmac_per_step"] * J_PER_GMAC["cpu" if accel == "cpu" else "npu"]
    parts = {
        "capture_J": CAPTURE_W[capture] * 3600,
        "gate_J": STEP_HZ * 3600 * e_gate,
        "heavy_J": heavy_per_hour * e_heavy,
        "store_J": events_per_hour * E_STORE_J,
    }
    tot = sum(parts.values())
    return {**parts, "total_J": tot, "avg_power_W": tot / 3600, "battery_pct_per_hour": tot / 3600 / BATTERY_WH * 100}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.parse_args()
    work = model_work()
    regions = regions_per_heavy_frame()
    pareto = json.loads((RESULTS / "m2" / "pareto_test.json").read_text())
    # operating points: always-heavy, uniform, learned trigger at each budget (seed 0), oracle
    ops = []
    for r in pareto["rows"]:
        if r["policy"] in ("uniform-k1",) or r["policy"].startswith(("uniform-k", "learned+imu", "oracle", "motion")):
            if r.get("seed") not in (None, 0):
                continue
            ops.append(r)
    rows = []
    for r in ops:
        for accel in J_PER_GMAC:
            for cap in CAPTURE_W:
                e = energy_per_hour(r["heavy_per_hour"], r["events_per_hour"], work, regions, accel, cap)
                rows.append(
                    {
                        "policy": r["policy"],
                        "target_budget": r.get("target_budget"),
                        "frac_processed": r["frac_processed"],
                        "heavy_per_hour": r["heavy_per_hour"],
                        "where_acc": r["metrics"]["where_acc"].get("mean"),
                        "accel": accel,
                        "capture": cap,
                        **e,
                    }
                )
    # break-even: max heavy invocations/hour within 5 %/h for each (accel, capture)
    budget_J = 0.05 * BATTERY_WH * 3600
    breakeven = {}
    for accel in J_PER_GMAC:
        for cap in CAPTURE_W:
            base = energy_per_hour(0, 0, work, regions, accel, cap)["total_J"]
            per = energy_per_hour(1, 0, work, regions, accel, cap)["total_J"] - base
            breakeven[f"{accel}/{cap}"] = {
                "max_heavy_per_hour": max(0.0, (budget_J - base) / per),
                "max_frac_of_2fps_frames": max(0.0, (budget_J - base) / per) / (STEP_HZ * 3600),
                "capture_alone_pct_per_hour": CAPTURE_W[cap] / BATTERY_WH * 100,
            }
    write_result(
        "m8/energy_model.json",
        {
            **result_header(
                "estimated",
                None,
                {"battery_Wh": BATTERY_WH, "capture_W": CAPTURE_W, "J_per_GMAC": J_PER_GMAC, "E_store_J": E_STORE_J},
            ),
            "WARNING": "estimated from an analytic model with assumed constants; not an on-device measurement",
            "work": work,
            "regions_per_frame": regions,
            "rows": rows,
            "breakeven_5pct": breakeven,
        },
    )
    for k, v in breakeven.items():
        print(k, {a: round(b, 3) for a, b in v.items()})


if __name__ == "__main__":
    main()
