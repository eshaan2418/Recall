"""M10 user-study analysis (pre-specified; runs unchanged on real or simulated data).

Real data:      ``python -m recall.study.analysis``  reads ``study/data/*.csv`` and writes aggregate results to
                ``results/m10/``. If ``study/data/trials.csv`` does not exist it prints a notice and exits 0.
Simulated data: ``python -m recall.study.analysis --simulated``  reads ``study/sim_output/data/`` and writes to
                ``study/sim_output/analysis/``; every output is stamped SIMULATED.

Pre-specified analyses
----------------------
Primary (Holm over the 2 contrasts Recall-vs-B2, Recall-vs-none):
    log(time_to_find_s) ~ condition + trial_c + (1|participant) + (1|item)
    linear mixed model with crossed random intercepts (statsmodels MixedLM, REML). Not-found trials enter at the
    time cap (conservative; a sensitivity analysis on found-only trials is also reported). Fallback if the crossed
    model fails: (1|participant) with item as a fixed effect. Robustness: paired t-test on per-participant mean
    log time.
Secondary (one Holm family):
    success ~ condition + trial_c, GEE (binomial, exchangeable, clusters = participants), contrasts as above;
        if any condition has 0 % or 100 % success (separation) -> Wilcoxon on per-participant success rates.
    Raw NASA-TLX (mean of 6 subscales): Friedman omnibus over 3 conditions + pairwise Wilcoxon signed-rank.
    SUS (0-100), trust (mean of 3 items), usefulness (mean of 2): Wilcoxon signed-rank Recall vs B2.
Effect sizes: time ratio exp(beta) with 95 % CI; odds ratio; matched-pairs rank-biserial r for Wilcoxon.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from recall.paths import RESULTS
from recall.study.design import STUDY_DIR
from recall.study.schema import SUS_ITEMS, TLX_SCALES, TRUST_ITEMS, USEFUL_ITEMS, SchemaError, read_table
from recall.study.simulate import SIM_BANNER, SIM_DIR

REAL_DATA_DIR = STUDY_DIR / "data"
REAL_OUT_DIR = RESULTS / "m10"
SIM_DATA_DIR = SIM_DIR / "data"
SIM_OUT_DIR = SIM_DIR / "analysis"
PRIMARY_CONTRASTS = (("recall", "b2"), ("recall", "none"))
ALL_PAIRS = (("recall", "b2"), ("recall", "none"), ("b2", "none"))


# ----------------------------------------------------------------------------- loading
def load_data(data_dir: Path) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    for table in ("trials", "questionnaires", "participants"):
        p = data_dir / f"{table}.csv"
        if p.exists():
            out[table] = read_table(p, table)
    if "trials" not in out:
        raise FileNotFoundError(data_dir / "trials.csv")
    return out


def _is_simulated(tables: dict[str, pd.DataFrame]) -> bool:
    return any(
        "simulated" in df.columns and (df["simulated"].fillna(0).astype(int) == 1).any() for df in tables.values()
    )


def _excluded_ids(tables: dict[str, pd.DataFrame]) -> set[str]:
    p = tables.get("participants")
    if p is None:
        return set()
    return set(p.loc[p["session_completed"].astype(int) == 0, "participant_id"].astype(str))


# ----------------------------------------------------------------------------- stats helpers
def holm(pvals: dict[str, float]) -> dict[str, float]:
    keys = [k for k, v in pvals.items() if v is not None and not math.isnan(v)]
    order = sorted(keys, key=lambda k: pvals[k])
    m = len(order)
    adj: dict[str, float] = {}
    running = 0.0
    for i, k in enumerate(order):
        running = max(running, min(1.0, (m - i) * pvals[k]))
        adj[k] = running
    return {k: adj.get(k, float("nan")) for k in pvals}


def wilcoxon_paired(x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    """Two-sided Wilcoxon signed-rank on paired samples + matched-pairs rank-biserial r (positive: x > y)."""
    d = np.asarray(x, float) - np.asarray(y, float)
    d = d[~np.isnan(d)]
    n = len(d)
    res: dict[str, float] = {"n": n, "median_diff": float(np.median(d)) if n else float("nan")}
    nz = d[d != 0]
    if len(nz) == 0:
        res.update(stat=float("nan"), p=1.0, r_rb=0.0)
        return res
    ranks = stats.rankdata(np.abs(nz))
    wp, wm = ranks[nz > 0].sum(), ranks[nz < 0].sum()
    res["r_rb"] = float((wp - wm) / (wp + wm))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        w = stats.wilcoxon(d, zero_method="wilcox", alternative="two-sided")
    res["stat"], res["p"] = float(w.statistic), float(w.pvalue)
    return res


def _dummies(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["is_recall"] = (df["condition"] == "recall").astype(float)
    df["is_b2"] = (df["condition"] == "b2").astype(float)
    df["trial_c"] = df["trial_index"].astype(float) - df["trial_index"].astype(float).mean()
    return df


def _contrast(params: pd.Series, cov: pd.DataFrame, a: str, b: str) -> tuple[float, float]:
    """Estimate and SE of (effect of a) - (effect of b), reference level 'none'."""
    vec = pd.Series(0.0, index=params.index)
    if a != "none":
        vec[f"is_{a}"] += 1.0
    if b != "none":
        vec[f"is_{b}"] -= 1.0
    est = float(vec @ params)
    se = float(math.sqrt(max(float(vec @ cov.loc[vec.index, vec.index] @ vec), 0.0)))
    return est, se


def _z_summary(est: float, se: float, transform=math.exp) -> dict[str, float]:
    z = est / se if se > 0 else float("nan")
    p = float(2 * stats.norm.sf(abs(z))) if se > 0 else float("nan")
    return {
        "estimate": est,
        "se": se,
        "z": z,
        "p": p,
        "ratio": transform(est),
        "ratio_lo": transform(est - 1.96 * se),
        "ratio_hi": transform(est + 1.96 * se),
    }


# ----------------------------------------------------------------------------- models
def fit_time_model(trials: pd.DataFrame, found_only: bool = False) -> dict[str, Any]:
    import statsmodels.formula.api as smf

    df = _dummies(trials)
    if found_only:
        df = df[df["success"] == 1]
    df["log_t"] = np.log(np.maximum(df["time_to_find_s"].astype(float), 0.5))
    df["grp"] = 1
    model = "crossed: (1|participant) + (1|item)"
    fit = None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            md = smf.mixedlm(
                "log_t ~ is_recall + is_b2 + trial_c",
                df,
                groups="grp",
                vc_formula={"participant": "0 + C(participant_id)", "item": "0 + C(item)"},
            )
            fit = md.fit(reml=True)
            if not np.all(np.isfinite(fit.bse[["is_recall", "is_b2"]])):
                fit = None
        except Exception:
            fit = None
        if fit is None:
            model = "fallback: (1|participant) + item fixed"
            md = smf.mixedlm("log_t ~ is_recall + is_b2 + trial_c + C(item)", df, groups="participant_id")
            fit = md.fit(reml=True)
    fe = ["Intercept", "is_recall", "is_b2", "trial_c"]
    params = fit.fe_params[fe]
    cov = fit.cov_params().loc[fe, fe]
    contrasts = {f"{a}_vs_{b}": _z_summary(*_contrast(params, cov, a, b)) for a, b in ALL_PAIRS}
    out: dict[str, Any] = {
        "model": model,
        "n_obs": len(df),
        "converged": bool(getattr(fit, "converged", True)),
        "trial_slope": float(params["trial_c"]),
        "contrasts": contrasts,
    }
    if "vcomp" in dir(fit) and model.startswith("crossed"):
        out["sd_random"] = {
            k: float(math.sqrt(max(v, 0))) for k, v in zip(["participant", "item"], np.atleast_1d(fit.vcomp))
        }
        out["sd_resid"] = float(math.sqrt(fit.scale))
    return out


def participant_level_time(trials: pd.DataFrame) -> dict[str, Any]:
    df = trials.assign(log_t=np.log(np.maximum(trials["time_to_find_s"].astype(float), 0.5)))
    wide = df.pivot_table(index="participant_id", columns="condition", values="log_t", aggfunc="mean")
    out = {}
    for a, b in ALL_PAIRS:
        if a in wide and b in wide:
            d = (wide[a] - wide[b]).dropna()
            t = stats.ttest_1samp(d, 0.0) if len(d) > 1 else None
            out[f"{a}_vs_{b}"] = {
                "n": len(d),
                "mean_log_diff": float(d.mean()),
                "ratio": float(np.exp(d.mean())),
                "p": float(t.pvalue) if t is not None else float("nan"),
            }
    return out


def fit_success_model(trials: pd.DataFrame) -> dict[str, Any]:
    rates = trials.groupby("condition")["success"].mean()
    separated = bool(((rates == 0) | (rates == 1)).any())
    out: dict[str, Any] = {"rates": {k: float(v) for k, v in rates.items()}, "separation": separated}
    if not separated:
        import statsmodels.api as sm
        import statsmodels.formula.api as smf

        df = _dummies(trials)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fit = smf.gee(
                    "success ~ is_recall + is_b2 + trial_c",
                    groups="participant_id",
                    data=df,
                    family=sm.families.Binomial(),
                    cov_struct=sm.cov_struct.Exchangeable(),
                ).fit()
            fe = ["Intercept", "is_recall", "is_b2", "trial_c"]
            params, cov = fit.params[fe], fit.cov_params().loc[fe, fe]
            out["method"] = "GEE binomial, exchangeable, robust SE"
            out["contrasts"] = {f"{a}_vs_{b}": _z_summary(*_contrast(params, cov, a, b)) for a, b in ALL_PAIRS}
            return out
        except Exception as e:  # pragma: no cover - numerical failure path
            out["gee_error"] = repr(e)
    wide = trials.pivot_table(index="participant_id", columns="condition", values="success", aggfunc="mean")
    out["method"] = "Wilcoxon signed-rank on per-participant success rates (separation/GEE failure fallback)"
    out["contrasts"] = {f"{a}_vs_{b}": wilcoxon_paired(wide[a].to_numpy(), wide[b].to_numpy()) for a, b in ALL_PAIRS}
    return out


def questionnaire_scores(q: pd.DataFrame) -> pd.DataFrame:
    q = q.copy()
    q["tlx_raw"] = q[[f"tlx_{s}" for s in TLX_SCALES]].astype(float).mean(axis=1)
    sus = q[[f"sus_{i}" for i in range(1, SUS_ITEMS + 1)]].astype(float)
    odd = sus[[f"sus_{i}" for i in range(1, SUS_ITEMS + 1, 2)]] - 1
    even = 5 - sus[[f"sus_{i}" for i in range(2, SUS_ITEMS + 1, 2)]]
    q["sus_score"] = (odd.sum(axis=1, min_count=5) + even.sum(axis=1, min_count=5)) * 2.5
    q["trust"] = q[[f"trust_{i}" for i in range(1, TRUST_ITEMS + 1)]].astype(float).mean(axis=1)
    q["useful"] = q[[f"useful_{i}" for i in range(1, USEFUL_ITEMS + 1)]].astype(float).mean(axis=1)
    return q[["participant_id", "condition", "tlx_raw", "sus_score", "trust", "useful"]]


def questionnaire_tests(q: pd.DataFrame) -> dict[str, Any]:
    s = questionnaire_scores(q)
    out: dict[str, Any] = {"descriptives": {}}
    for m in ("tlx_raw", "sus_score", "trust", "useful"):
        g = s.dropna(subset=[m]).groupby("condition")[m]
        out["descriptives"][m] = {
            c: {"median": float(v.median()), "mean": float(v.mean()), "n": int(v.size)} for c, v in g
        }
    tlx = s.pivot_table(index="participant_id", columns="condition", values="tlx_raw").dropna()
    if tlx.shape[1] == 3 and len(tlx) >= 2:
        fr = stats.friedmanchisquare(tlx["recall"], tlx["none"], tlx["b2"])
        out["tlx_friedman"] = {"chi2": float(fr.statistic), "p": float(fr.pvalue), "n": len(tlx)}
    out["tests"] = {}
    for a, b in ALL_PAIRS:
        if a in tlx and b in tlx:
            out["tests"][f"tlx_{a}_vs_{b}"] = wilcoxon_paired(tlx[a].to_numpy(), tlx[b].to_numpy())
    for m in ("sus_score", "trust", "useful"):
        w = s.pivot_table(index="participant_id", columns="condition", values=m)
        if "recall" in w and "b2" in w:
            w = w[["recall", "b2"]].dropna()
            out["tests"][f"{m}_recall_vs_b2"] = wilcoxon_paired(w["recall"].to_numpy(), w["b2"].to_numpy())
    return out


def reliance_descriptives(trials: pd.DataFrame) -> dict[str, Any]:
    """Appropriate reliance: how often participants followed correct vs incorrect assistant answers."""
    out: dict[str, Any] = {}
    for c in ("recall", "b2"):
        d = trials[(trials["condition"] == c)].dropna(subset=["assistant_correct", "followed_assistant_first"])
        if d.empty:
            continue
        r: dict[str, Any] = {"assistant_accuracy": float(d["assistant_correct"].astype(float).mean()), "n": len(d)}
        for corr in (1, 0):
            sub = d[d["assistant_correct"].astype(int) == corr]
            key = "followed_when_correct" if corr else "followed_when_wrong"
            r[key] = float(sub["followed_assistant_first"].astype(float).mean()) if len(sub) else float("nan")
        out[c] = r
    rec = trials[(trials["condition"] == "recall")].dropna(subset=["assistant_confidence", "assistant_correct"])
    if len(rec) >= 3 and rec["assistant_correct"].nunique() == 2:
        out["recall_confidence_auc"] = float(
            stats.mannwhitneyu(
                rec.loc[rec["assistant_correct"].astype(int) == 1, "assistant_confidence"],
                rec.loc[rec["assistant_correct"].astype(int) == 0, "assistant_confidence"],
            ).statistic
            / ((rec["assistant_correct"].astype(int) == 1).sum() * (rec["assistant_correct"].astype(int) == 0).sum())
        )
    return out


# ----------------------------------------------------------------------------- orchestration
def analyse(tables: dict[str, pd.DataFrame]) -> dict[str, Any]:
    excl = _excluded_ids(tables)
    trials = tables["trials"][~tables["trials"]["participant_id"].isin(excl)].copy()
    res: dict[str, Any] = {
        "n_participants": int(trials["participant_id"].nunique()),
        "n_trials": len(trials),
        "excluded_incomplete": sorted(excl),
        "descriptives": {
            c: {
                "median_time_s": float(g["time_to_find_s"].median()),
                "success_rate": float(g["success"].mean()),
                "n_trials": len(g),
            }
            for c, g in trials.groupby("condition")
        },
    }
    res["time"] = fit_time_model(trials)
    if trials["success"].sum() > 10:
        res["time_found_only"] = fit_time_model(trials, found_only=True)
    res["time_participant_level"] = participant_level_time(trials)
    prim = holm({f"{a}_vs_{b}": res["time"]["contrasts"][f"{a}_vs_{b}"]["p"] for a, b in PRIMARY_CONTRASTS})
    for k, v in prim.items():
        res["time"]["contrasts"][k]["p_holm"] = v
    res["success"] = fit_success_model(trials)
    sec: dict[str, float] = {
        f"success_{a}_vs_{b}": res["success"]["contrasts"][f"{a}_vs_{b}"]["p"] for a, b in PRIMARY_CONTRASTS
    }
    if "questionnaires" in tables:
        q = tables["questionnaires"][~tables["questionnaires"]["participant_id"].isin(excl)]
        res["questionnaires"] = questionnaire_tests(q)
        sec.update({k: v["p"] for k, v in res["questionnaires"]["tests"].items()})
    res["secondary_holm"] = holm(sec)
    res["reliance"] = reliance_descriptives(trials)
    return res


def _fmt_p(p: float) -> str:
    return "n/a" if p is None or (isinstance(p, float) and math.isnan(p)) else ("<0.001" if p < 0.001 else f"{p:.3f}")


def render_report(res: dict[str, Any], simulated: bool) -> str:
    banner = f"> **{SIM_BANNER}**\n\n" if simulated else ""
    L = [f"# M10 user study — analysis{' (SIMULATED)' if simulated else ''}", "", banner.rstrip(), ""]
    L += [
        f"Participants analysed: {res['n_participants']} · trials: {res['n_trials']}"
        + (f" · excluded (incomplete): {', '.join(res['excluded_incomplete'])}" if res["excluded_incomplete"] else ""),
        "",
        "## Descriptives",
        "",
        "| condition | median time-to-find (s) | success rate | trials |",
        "|---|---|---|---|",
    ]
    for c, d in res["descriptives"].items():
        L.append(f"| {c} | {d['median_time_s']:.1f} | {d['success_rate']:.2f} | {d['n_trials']} |")
    t = res["time"]
    L += [
        "",
        "## Primary: log time-to-find (LMM)",
        "",
        f"Model: `log_t ~ condition + trial_c`, random effects {t['model']}; n_obs = {t['n_obs']}; "
        f"trial slope = {t['trial_slope']:+.3f} log-s/trial. Not-found trials enter at the time cap.",
        "",
        "| contrast | time ratio | 95% CI | p (z) | p (Holm, primary) |",
        "|---|---|---|---|---|",
    ]
    for k, c in t["contrasts"].items():
        L.append(
            f"| {k} | {c['ratio']:.2f} | [{c['ratio_lo']:.2f}, {c['ratio_hi']:.2f}] | {_fmt_p(c['p'])} | "
            f"{_fmt_p(c.get('p_holm', float('nan')))} |"
        )
    if "time_found_only" in res:
        L += ["", "Sensitivity (found trials only):", ""]
        for k, c in res["time_found_only"]["contrasts"].items():
            L.append(f"- {k}: ratio {c['ratio']:.2f} [{c['ratio_lo']:.2f}, {c['ratio_hi']:.2f}], p = {_fmt_p(c['p'])}")
    L += ["", "Robustness (paired t on per-participant mean log time):", ""]
    for k, c in res["time_participant_level"].items():
        L.append(f"- {k}: ratio {c['ratio']:.2f}, p = {_fmt_p(c['p'])} (n = {c['n']})")
    s = res["success"]
    L += ["", "## Secondary (Holm-corrected as one family)", "", f"Success: {s['method']}.", ""]
    hol = res["secondary_holm"]
    L += ["| test | effect | p | p (Holm) |", "|---|---|---|---|"]
    for a, b in PRIMARY_CONTRASTS:
        c = s["contrasts"][f"{a}_vs_{b}"]
        eff = f"OR {c['ratio']:.2f}" if "ratio" in c else f"r_rb {c['r_rb']:+.2f}"
        L.append(f"| success {a} vs {b} | {eff} | {_fmt_p(c['p'])} | {_fmt_p(hol[f'success_{a}_vs_{b}'])} |")
    if "questionnaires" in res:
        qq = res["questionnaires"]
        for k, c in qq["tests"].items():
            L.append(
                f"| {k} | median diff {c['median_diff']:+.1f}, r_rb {c['r_rb']:+.2f} | {_fmt_p(c['p'])} | "
                f"{_fmt_p(hol[k])} |"
            )
        if "tlx_friedman" in qq:
            f = qq["tlx_friedman"]
            L += ["", f"NASA-TLX Friedman omnibus: chi2 = {f['chi2']:.2f}, p = {_fmt_p(f['p'])}, n = {f['n']}."]
        L += ["", "| measure | condition | median | mean | n |", "|---|---|---|---|---|"]
        for m, byc in qq["descriptives"].items():
            for c, d in byc.items():
                L.append(f"| {m} | {c} | {d['median']:.1f} | {d['mean']:.1f} | {d['n']} |")
    if res.get("reliance"):
        L += ["", "## Reliance (descriptive)", "", "```", json.dumps(res["reliance"], indent=1), "```"]
    L += [
        "",
        "Notes: z-based p-values from MixedLM/GEE are anti-conservative at n ~ 12; the participant-level paired "
        "analysis is the small-sample check. Lower TLX = less workload; higher SUS/trust/usefulness = better.",
    ]
    if simulated:
        L += ["", f"**{SIM_BANNER}**"]
    return "\n".join(x for x in L if x is not None) + "\n"


def _jsonable(o: Any) -> Any:
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, list | tuple):
        return [_jsonable(v) for v in o]
    if isinstance(o, np.integer | np.bool_):
        return o.item()
    if isinstance(o, float | np.floating):
        return None if math.isnan(float(o)) else float(o)
    return o


def run(data_dir: Path, out_dir: Path, simulated: bool) -> dict[str, Any]:
    tables = load_data(data_dir)
    has_sim = _is_simulated(tables)
    if not simulated and has_sim:
        raise SchemaError(f"{data_dir} contains rows marked simulated=1; refusing to analyse them as real data")
    if simulated and not has_sim:
        raise SchemaError("--simulated was given but the data are not marked simulated=1")
    if simulated and out_dir.resolve().is_relative_to(RESULTS.resolve()):
        raise ValueError("simulated analyses must not be written under results/; use study/sim_output/")
    res = analyse(tables)
    res["simulated"] = simulated
    if simulated:
        res["WARNING"] = SIM_BANNER
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "analysis.json").write_text(json.dumps(_jsonable(res), indent=1))
    (out_dir / "report.md").write_text(render_report(res, simulated))
    return res


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--simulated", action="store_true", help="analyse study/sim_output/data (SIMULATED)")
    ap.add_argument("--data-dir", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)
    data_dir = args.data_dir or (SIM_DATA_DIR if args.simulated else REAL_DATA_DIR)
    out_dir = args.out or (SIM_OUT_DIR if args.simulated else REAL_OUT_DIR)
    if not (data_dir / "trials.csv").exists():
        if args.simulated:
            print(f"no simulated data in {data_dir}; run `python -m recall.study.simulate` first", file=sys.stderr)
            return 1
        print(f"No study data yet: {data_dir / 'trials.csv'} not found. Nothing to analyse (this is not an error).")
        print("To validate the pipeline on SIMULATED participants: make study-sim")
        return 0
    run(data_dir, out_dir, args.simulated)
    print((out_dir / "report.md").read_text())
    print(f"wrote {out_dir / 'report.md'} and {out_dir / 'analysis.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
