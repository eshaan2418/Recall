"""SIMULATED participants for validating the M10 analysis pipeline.

!!! Everything produced here is SIMULATED. It exists only to check that the analysis code runs and recovers
!!! planted effects, and to give a simulation-based sensitivity estimate. It is NOT a study result and must
!!! never be reported as one. Outputs go to ``study/sim_output/`` and every row carries ``simulated = 1``.

Generative model (all parameters are assumptions, not measurements):
    log T_pit = log(median_none) + beta_cond + slope * (trial - 1) + u_p + v_i + e_pit,
    u_p ~ N(0, sd_participant), v_i ~ N(0, sd_item), e ~ N(0, sd_resid);
    trials with T > cap are failures (success = 0, time recorded as the cap).
Questionnaires are drawn from participant-level latent scores plus condition shifts.

Usage: ``python -m recall.study.simulate [--n 12] [--seed 0] [--out study/sim_output/data]``
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from recall.paths import RESULTS
from recall.study.design import STUDY_DIR, StudyConfig, generate_assignments, load_config
from recall.study.schema import SUS_ITEMS, TLX_SCALES, TRUST_ITEMS, USEFUL_ITEMS

SIM_DIR = STUDY_DIR / "sim_output"
SIM_BANNER = "SIMULATED DATA — pipeline validation only. NOT study results; do not report."


@dataclass
class SimParams:
    median_none_s: float = 60.0
    # planted log time ratios vs no assistant (negative = faster)
    log_ratio: dict[str, float] = field(
        default_factory=lambda: {"none": 0.0, "b2": math.log(0.80), "recall": math.log(0.55)}
    )
    trial_slope: float = -0.02
    sd_participant: float = 0.35
    sd_item: float = 0.25
    sd_resid: float = 0.55
    assistant_acc: dict[str, float] = field(default_factory=lambda: {"recall": 0.85, "b2": 0.70})
    tlx_shift: dict[str, float] = field(default_factory=lambda: {"none": 0.0, "b2": -5.0, "recall": -12.0})
    sus_shift: dict[str, float] = field(default_factory=lambda: {"b2": 0.0, "recall": 8.0})
    trust_shift: dict[str, float] = field(default_factory=lambda: {"b2": 0.0, "recall": 0.8})
    useful_shift: dict[str, float] = field(default_factory=lambda: {"b2": 0.0, "recall": 0.8})


def simulate_trials(
    cfg: StudyConfig, params: SimParams, rng: np.random.Generator, assign: pd.DataFrame | None = None
) -> pd.DataFrame:
    a = generate_assignments(cfg) if assign is None else assign
    pids = sorted(a["participant_id"].unique())
    u = dict(zip(pids, rng.normal(0, params.sd_participant, len(pids))))
    v = dict(zip(cfg.items, rng.normal(0, params.sd_item, len(cfg.items))))
    n = len(a)
    cond = a["condition"].to_numpy()
    logt = (
        math.log(params.median_none_s)
        + np.array([params.log_ratio[c] for c in cond])
        + params.trial_slope * (a["trial_index"].to_numpy() - 1)
        + a["participant_id"].map(u).to_numpy()
        + a["item"].map(v).to_numpy()
        + rng.normal(0, params.sd_resid, n)
    )
    t = np.exp(logt)
    success = (t <= cfg.time_cap_s).astype(int)
    t = np.minimum(t, cfg.time_cap_s).round(1)
    is_asst = np.isin(cond, ("recall", "b2"))
    acc = np.array([params.assistant_acc.get(c, np.nan) for c in cond])
    correct = np.where(is_asst, (rng.random(n) < acc).astype(float), np.nan)
    conf = np.where(correct == 1, rng.beta(8, 2, n), rng.beta(3, 4, n))
    conf = np.where(cond == "recall", conf.round(2), np.nan)
    p_follow = np.where(cond == "recall", np.where(conf > 0.5, 0.85, 0.4), 0.7)
    follow = np.where(is_asst, (rng.random(n) < p_follow).astype(float), np.nan)
    nq = np.where(is_asst, 1 + rng.poisson(0.4, n), 0)
    return pd.DataFrame(
        {
            "participant_id": a["participant_id"],
            "trial_index": a["trial_index"],
            "block_index": a["block_index"],
            "condition": cond,
            "item": a["item"],
            "hiding_place": a["hiding_place"],
            "time_to_find_s": t,
            "success": success,
            "n_queries": nq,
            "assistant_correct": pd.array(correct, dtype="Int64"),
            "assistant_confidence": conf,
            "followed_assistant_first": pd.array(follow, dtype="Int64"),
            "simulated": 1,
        }
    )


def _clip_round(x: np.ndarray, lo: int, hi: int, step: int = 1) -> np.ndarray:
    return (np.clip(np.round(x / step) * step, lo, hi)).astype(int)


def simulate_questionnaires(assign: pd.DataFrame, params: SimParams, rng: np.random.Generator) -> pd.DataFrame:
    blocks = assign.drop_duplicates(["participant_id", "block_index"])[["participant_id", "condition", "block_index"]]
    pids = sorted(blocks["participant_id"].unique())
    tlx_base = dict(zip(pids, rng.normal(50, 12, len(pids))))
    sus_base = dict(zip(pids, rng.normal(68, 10, len(pids))))
    trust_base = dict(zip(pids, rng.normal(4.5, 0.9, len(pids))))
    useful_base = dict(zip(pids, rng.normal(4.8, 0.9, len(pids))))
    rows = []
    for _, b in blocks.iterrows():
        pid, c = b["participant_id"], b["condition"]
        r: dict = {"participant_id": pid, "condition": c, "block_index": int(b["block_index"])}
        lvl = tlx_base[pid] + params.tlx_shift[c] + rng.normal(0, 6)
        for s in TLX_SCALES:
            r[f"tlx_{s}"] = int(_clip_round(np.array(lvl + rng.normal(0, 10)), 0, 100, 5))
        if c in ("recall", "b2"):
            sus = sus_base[pid] + params.sus_shift[c] + rng.normal(0, 6)
            frac = np.clip(sus / 100.0, 0, 1)
            for i in range(1, SUS_ITEMS + 1):
                raw = 1 + 4 * frac if i % 2 == 1 else 5 - 4 * frac
                r[f"sus_{i}"] = int(_clip_round(np.array(raw + rng.normal(0, 0.6)), 1, 5))
            tr = trust_base[pid] + params.trust_shift[c] + rng.normal(0, 0.4)
            for i in range(1, TRUST_ITEMS + 1):
                r[f"trust_{i}"] = int(_clip_round(np.array(tr + rng.normal(0, 0.6)), 1, 7))
            us = useful_base[pid] + params.useful_shift[c] + rng.normal(0, 0.4)
            for i in range(1, USEFUL_ITEMS + 1):
                r[f"useful_{i}"] = int(_clip_round(np.array(us + rng.normal(0, 0.6)), 1, 7))
        r["simulated"] = 1
        rows.append(r)
    df = pd.DataFrame(rows)
    for col in df.columns:
        if col.startswith(("sus_", "trust_", "useful_")):
            df[col] = df[col].astype("Int64")
    return df


def simulate_participants(assign: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    p = assign.drop_duplicates("participant_id")[["participant_id", "design_row"]].reset_index(drop=True)
    bands = np.array(["18-24", "25-34", "35-44", "45-54", "55-64", "65+"])
    p["age_band"] = rng.choice(bands, len(p), p=[0.25, 0.35, 0.15, 0.1, 0.1, 0.05])
    p["tech_familiarity"] = rng.integers(2, 6, len(p))
    p["familiar_with_space"] = 0
    p["session_completed"] = 1
    p["simulated"] = 1
    return p


def simulate_study(cfg: StudyConfig, params: SimParams | None = None, seed: int = 0) -> dict[str, pd.DataFrame]:
    params = params or SimParams()
    rng = np.random.default_rng(seed)
    assign = generate_assignments(cfg)
    return {
        "participants": simulate_participants(assign, rng),
        "trials": simulate_trials(cfg, params, rng, assign),
        "questionnaires": simulate_questionnaires(assign, params, rng),
    }


def write_simulated(tables: dict[str, pd.DataFrame], out_dir: Path) -> None:
    if out_dir.resolve().is_relative_to(RESULTS.resolve()):
        raise ValueError("simulated data must not be written under results/; use study/sim_output/")
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, df in tables.items():
        df.to_csv(out_dir / f"{name}.csv", index=False)
    (out_dir / "README_SIMULATED.txt").write_text(
        SIM_BANNER + "\nGenerated by recall.study.simulate. Every row has simulated=1.\n"
    )


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=SIM_DIR / "data")
    args = ap.parse_args(argv)
    cfg = load_config()
    cfg.n_participants = args.n
    tables = simulate_study(cfg, seed=args.seed)
    write_simulated(tables, args.out)
    print(f"{SIM_BANNER}\nwrote simulated CSVs for n={args.n} -> {args.out}")


if __name__ == "__main__":
    main()
