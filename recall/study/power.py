"""SIMULATION-BASED power / sensitivity for the M10 user study.

!!! SIMULATED. Power depends entirely on the assumed variance components in ``SimParams`` (participant SD,
!!! item SD, residual SD, SUS spread), which are guesses, not measurements. Re-run with variance estimates from
!!! a pilot before relying on these numbers. Outputs go to ``study/sim_output/power.{md,json}``.

For each n and each planted Recall-vs-B2 time ratio, simulate the full design, fit the pre-specified LMM and
count how often the Recall-vs-B2 contrast is significant (two-sided alpha = .05) in the planted direction. The
same is done for the small-sample participant-level paired t-test and for SUS (Wilcoxon, Recall vs B2).

Usage: ``python -m recall.study.power [--reps 200] [--n 10 12 15] [--jobs -1]``
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import replace

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from recall.study.analysis import fit_time_model, participant_level_time, questionnaire_scores, wilcoxon_paired
from recall.study.design import generate_assignments, load_config
from recall.study.simulate import SIM_BANNER, SIM_DIR, SimParams, simulate_questionnaires, simulate_trials

TIME_RATIOS = (1.0, 0.9, 0.8, 0.75, 0.7, 0.65, 0.6, 0.5)  # Recall time / B2 time
SUS_DIFFS = (0.0, 4.0, 8.0, 12.0, 16.0)  # SUS points, Recall - B2


def _one_time(n: int, ratio: float, rep: int, base: SimParams) -> tuple[bool, bool]:
    cfg = load_config()
    cfg.n_participants = n
    cfg.seed = 1000 + rep  # new randomization per replicate
    lr = dict(base.log_ratio)
    lr["recall"] = lr["b2"] + math.log(ratio)
    p = replace(base, log_ratio=lr)
    rng = np.random.default_rng([n, int(ratio * 1000), rep])
    t = simulate_trials(cfg, p, rng)
    c = fit_time_model(t)["contrasts"]["recall_vs_b2"]
    lmm_hit = c["p"] < 0.05 and c["estimate"] < 0 if ratio < 1 else c["p"] < 0.05
    pl = participant_level_time(t)["recall_vs_b2"]
    pl_hit = pl["p"] < 0.05 and pl["mean_log_diff"] < 0 if ratio < 1 else pl["p"] < 0.05
    return bool(lmm_hit), bool(pl_hit)


def _one_sus(n: int, diff: float, rep: int, base: SimParams) -> bool:
    cfg = load_config()
    cfg.n_participants = n
    cfg.seed = 2000 + rep
    a = generate_assignments(cfg)
    p = replace(base, sus_shift={"b2": 0.0, "recall": diff})
    q = simulate_questionnaires(a, p, np.random.default_rng([n, int(diff * 10), rep, 7]))
    s = questionnaire_scores(q).pivot_table(index="participant_id", columns="condition", values="sus_score")
    w = wilcoxon_paired(s["recall"].to_numpy(), s["b2"].to_numpy())
    return bool(w["p"] < 0.05 and (w["median_diff"] > 0 or diff == 0))


def mde(power_by_effect: dict[float, float], target: float = 0.8, smaller_is_bigger: bool = True) -> float | None:
    """Smallest effect (first in order of increasing effect size) reaching the target power."""
    keys = sorted(power_by_effect, reverse=smaller_is_bigger)
    for k in keys:
        if power_by_effect[k] >= target and k != (1.0 if smaller_is_bigger else 0.0):
            return k
    return None


def run(ns: list[int], reps: int, jobs: int) -> dict:
    base = SimParams()
    t0 = time.time()
    out: dict = {
        "WARNING": SIM_BANNER,
        "assumptions": {
            "sd_participant": base.sd_participant,
            "sd_item": base.sd_item,
            "sd_resid": base.sd_resid,
            "median_none_s": base.median_none_s,
            "b2_vs_none_ratio": math.exp(base.log_ratio["b2"]),
            "items_per_condition": load_config().items_per_condition,
            "reps": reps,
            "alpha": 0.05,
            "sus_participant_sd": 10,
            "sus_within_noise_sd": 6,
        },
        "time": {},
        "sus": {},
    }
    for n in ns:
        tasks = [(r, k) for r in TIME_RATIOS for k in range(reps)]
        hits = Parallel(n_jobs=jobs)(delayed(_one_time)(n, r, k, base) for r, k in tasks)
        df = pd.DataFrame([(r, h[0], h[1]) for (r, _), h in zip(tasks, hits)], columns=["ratio", "lmm", "paired"])
        g = df.groupby("ratio").mean()
        out["time"][n] = {
            "lmm_power": {float(r): float(v) for r, v in g["lmm"].items()},
            "paired_t_power": {float(r): float(v) for r, v in g["paired"].items()},
        }
        out["time"][n]["lmm_mde_ratio"] = mde(out["time"][n]["lmm_power"])
        out["time"][n]["paired_t_mde_ratio"] = mde(out["time"][n]["paired_t_power"])
        stasks = [(d, k) for d in SUS_DIFFS for k in range(reps)]
        shits = Parallel(n_jobs=jobs)(delayed(_one_sus)(n, d, k, base) for d, k in stasks)
        sdf = pd.DataFrame([(d, h) for (d, _), h in zip(stasks, shits)], columns=["diff", "hit"])
        sp = {float(d): float(v) for d, v in sdf.groupby("diff")["hit"].mean().items()}
        out["sus"][n] = {"wilcoxon_power": sp, "mde_points": mde(sp, smaller_is_bigger=False)}
    out["seconds"] = round(time.time() - t0, 1)
    return out


def render(res: dict) -> str:
    a = res["assumptions"]
    L = [
        "# M10 power / sensitivity — SIMULATION-BASED",
        "",
        f"> **{SIM_BANNER}**",
        "> These are properties of the analysis under ASSUMED variance components, not findings about Recall.",
        "",
        f"Assumptions: participant SD {a['sd_participant']}, item SD {a['sd_item']}, residual SD {a['sd_resid']} "
        f"(log-seconds); {a['items_per_condition']} items per condition; B2/none time ratio {a['b2_vs_none_ratio']:.2f}; "
        f"SUS between-participant SD {a['sus_participant_sd']}, within-participant noise SD "
        f"{a['sus_within_noise_sd']}; {a['reps']} replicates per cell; alpha = {a['alpha']} two-sided, unadjusted.",
        "",
        "## Time-to-find, Recall vs B2 (power to detect a planted time ratio)",
        "",
    ]
    ratios = sorted(TIME_RATIOS, reverse=True)
    L += ["| n | test | " + " | ".join(f"ratio {r:.2f}" for r in ratios) + " | MDE (80%) |"]
    L += ["|---|---|" + "---|" * len(ratios) + "---|"]
    for n, d in res["time"].items():
        for key, lab, mk in (
            ("lmm_power", "LMM", "lmm_mde_ratio"),
            ("paired_t_power", "paired t", "paired_t_mde_ratio"),
        ):
            cells = " | ".join(f"{d[key][float(r)]:.2f}" for r in ratios)
            m = d[mk]
            L.append(f"| {n} | {lab} | {cells} | {'none in grid' if m is None else f'{m:.2f}'} |")
    L += [
        "",
        "Ratio 1.0 column = false-positive rate (should be about 0.05).",
        "",
        "## SUS, Recall vs B2 (Wilcoxon)",
        "",
    ]
    L += [
        "| n | " + " | ".join(f"+{d:g} pts" for d in SUS_DIFFS) + " | MDE (80%) |",
        "|---|" + "---|" * (len(SUS_DIFFS) + 1),
    ]
    for n, d in res["sus"].items():
        cells = " | ".join(f"{d['wilcoxon_power'][float(x)]:.2f}" for x in SUS_DIFFS)
        m = d["mde_points"]
        L.append(f"| {n} | {cells} | {'none in grid' if m is None else f'+{m:g}'} |")
    L += ["", f"Runtime {res['seconds']} s.", "", f"**{SIM_BANNER}**", ""]
    return "\n".join(L)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reps", type=int, default=200)
    ap.add_argument("--n", type=int, nargs="+", default=[10, 12, 15])
    ap.add_argument("--jobs", type=int, default=-1)
    args = ap.parse_args(argv)
    res = run(args.n, args.reps, args.jobs)
    SIM_DIR.mkdir(parents=True, exist_ok=True)
    (SIM_DIR / "power.json").write_text(
        json.dumps(
            {
                **res,
                "time": {str(k): v for k, v in res["time"].items()},
                "sus": {str(k): v for k, v in res["sus"].items()},
            },
            indent=1,
        )
    )
    (SIM_DIR / "power.md").write_text(render(res))
    print(render(res))


if __name__ == "__main__":
    main()
