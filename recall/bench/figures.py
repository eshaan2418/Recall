"""Regenerate every paper/README figure from results/ (no figure is drawn from numbers typed by hand).

    python -m recall.bench.figures [--allow-missing]

Writes results/<milestone>/figs/*.png and copies them to paper/figs/.
Palette: validated categorical slots (blue, orange, aqua; ≤3 series per chart) on a light surface; text in ink
tokens; recessive grid; direct labels on lines.
"""

from __future__ import annotations

import argparse
import json
import shutil
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from recall.paths import REPO, RESULTS

S1, S2, S3, S4 = "#2a78d6", "#eb6834", "#1baf7a", "#6250d6"
INK, INK2, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e1", "#fcfcfb"
plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "axes.edgecolor": MUTED, "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2, "text.color": INK, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False, "font.size": 9,
    "lines.linewidth": 2, "lines.markersize": 5, "legend.frameon": False, "savefig.dpi": 200,
})  # fmt: skip

OUT: list[Path] = []


def _load(rel: str) -> dict | None:
    p = RESULTS / rel
    return json.loads(p.read_text()) if p.exists() else None


def _save(fig, rel: str) -> None:
    p = RESULTS / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(p)
    plt.close(fig)
    OUT.append(p)


def _m(d: dict | None, key: str) -> tuple[float, float, float]:
    x = (d or {}).get(key) or {}
    return x.get("mean", np.nan), x.get("lo", np.nan), x.get("hi", np.nan)


def fig_m1() -> None:
    r = _load("m1/systems_test.json")
    if not r:
        return
    names = [n for n in r["systems"] if not n.startswith("B3") or n == "B3-manual-q0.5"]
    b1 = _load("m1/b1_vlm_K16.json")
    fig, ax = plt.subplots(figsize=(6.4, 3.0))
    y = np.arange(len(names) + (1 if b1 else 0))
    vals = [_m(r["systems"][n]["metrics"], "where_acc") for n in names]
    if b1:
        names = [*names, f"B1-{b1['config'].get('model', 'VLM').split('/')[-1]} (subsample)"]
        vals.append(_m(b1["metrics"], "where_acc"))
    m = np.array([v[0] for v in vals])
    err = np.array([[v[0] - v[1], v[2] - v[0]] for v in vals]).T
    colors = [S1 if n.startswith("Recall") else MUTED for n in names]
    ax.barh(y, m, xerr=err, color=colors, height=0.6, error_kw={"ecolor": INK2, "lw": 1})
    ax.set_yticks(y, names)
    ax.invert_yaxis()
    ax.set_xlabel('"Where is X?" accuracy (test, 95% cluster-bootstrap CI)')
    for yi, mi in zip(y, m, strict=True):
        if np.isfinite(mi):
            ax.text(mi + 0.01, float(yi), f"{mi:.2f}", va="center", color=INK2, fontsize=8)
    _save(fig, "m1/figs/where_acc.png")


def fig_m2() -> None:
    r = _load("m2/pareto_test.json")
    if not r:
        return
    groups: dict[str, list] = defaultdict(list)
    for row in r["rows"]:
        p = row["policy"]
        fam = ("uniform" if p.startswith("uniform-k") else "learned (img+IMU)" if p == "learned+imu"
               else "learned (img)" if p == "learned" else p)  # fmt: skip
        groups[fam].append(row)
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.0), sharex=True)
    for ax, key, lab in ((axes[0], "where_acc", '"Where is X?" accuracy'), (axes[1], "temporal_acc", "temporal acc.")):
        for fam, col in (("uniform", MUTED), ("motion", S2), ("learned (img+IMU)", S1)):
            rows = groups.get(fam, [])
            if not rows:
                continue
            agg: dict[float, list] = defaultdict(list)
            for row in rows:
                agg[round(row["frac_processed"], 3) if fam == "uniform" else row["target_budget"]].append(row)
            xs, ys = [], []
            for _, rr in sorted(agg.items()):
                xs.append(np.mean([q["frac_processed"] for q in rr]))
                ys.append(np.mean([q["metrics"][key].get("mean", np.nan) for q in rr]))
            o = np.argsort(xs)
            ax.plot(np.array(xs)[o], np.array(ys)[o], "-o", color=col, label=fam)
        orc = groups.get("oracle")
        if orc:
            ax.scatter([orc[0]["frac_processed"]], [orc[0]["metrics"][key]["mean"]], marker="*", s=90, color=S3,
                       zorder=5, label="oracle trigger (B4)")  # fmt: skip
        ax.set_xscale("log")
        ax.set_xlabel("fraction of 2 fps frames sent to heavy perception")
        ax.set_ylabel(lab)
    axes[0].legend(fontsize=8, loc="lower right")
    _save(fig, "m2/figs/pareto.png")


def fig_m3() -> None:
    r = _load("m3/memory_storage.json")
    if not r:
        return
    rows = [x for x in r["rows"] if x["variant"] != "_summary"]
    fig, ax = plt.subplots(figsize=(5.6, 3.0))
    for i, pol in enumerate(sorted({x["policy"] for x in rows})):
        rr = [x for x in rows if x["policy"] == pol]
        for x in rr:
            acc = (x.get("metrics") or {}).get("where_acc", {}).get("mean")
            mk = "o" if acc is not None else "|"
            ax.scatter(x["projected_GB_per_week"], acc if acc is not None else 0, color=[S1, S2][i % 2], marker=mk)
            ax.annotate(f"{x['variant']}", (x["projected_GB_per_week"], acc if acc is not None else 0),
                        fontsize=7, color=INK2, xytext=(3, 3), textcoords="offset points")  # fmt: skip
    ax.axvline(1.0, color=MUTED, ls="--", lw=1)
    ax.text(1.0, ax.get_ylim()[1], " 1 GB/week budget", color=INK2, fontsize=8, va="top")
    ax.set_xscale("log")
    ax.set_xlabel("projected storage, GB/week (16 recorded h/day assumption)")
    ax.set_ylabel('"Where is X?" accuracy (no-acc variants at 0)')
    _save(fig, "m3/figs/storage.png")


def fig_m4() -> None:
    r = _load("m4/calibration_test.json")
    if not r:
        return
    sysname = next((n for n in r["systems"] if n.startswith("Recall-learned")), next(iter(r["systems"])))
    conf = r["systems"][sysname]["confidence"]
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.0))
    ax = axes[0]
    ax.plot([0, 1], [0, 1], color=MUTED, lw=1, ls="--")
    for name, col in (("always-confident", MUTED), ("raw-score", S2), ("isotonic", S1)):
        if name not in conf:
            continue
        rel = [b for b in conf[name]["reliability"] if b.get("n", 0) > 0]
        ax.plot([b["conf"] for b in rel], [b["acc"] for b in rel], "-o", color=col,
                label=f"{name} (ECE {conf[name]['ece']:.3f})")  # fmt: skip
    ax.set_xlabel("predicted confidence")
    ax.set_ylabel("observed accuracy")
    ax.legend(fontsize=7)
    ax = axes[1]
    for name, col in (("raw-score", S2), ("isotonic", S1)):
        if name in conf:
            rc = conf[name]["risk_coverage"]
            ax.plot(rc["coverage"], rc["sel_acc"], color=col, label=name)
    b2 = r["systems"].get("B2-frame-CLIP-2s")
    if b2 and "isotonic" in b2["confidence"]:
        rc = b2["confidence"]["isotonic"]["risk_coverage"]
        ax.plot(rc["coverage"], rc["sel_acc"], color=S3, label="B2 frame+CLIP (isotonic)")
    ax.set_xlabel("coverage (fraction answered)")
    ax.set_ylabel("selective accuracy")
    ax.legend(fontsize=7)
    _save(fig, "m4/figs/calibration.png")


def fig_m8() -> None:
    r = _load("m8/energy_model.json")
    if not r:
        return
    fig, ax = plt.subplots(figsize=(5.6, 3.0))
    for accel, col in (("npu", S1), ("gpu", S3), ("cpu", S2)):
        rows = sorted([x for x in r["rows"] if x["accel"] == accel and x["capture"] == "low"
                       and x["policy"].startswith(("uniform-k", "learned+imu", "oracle"))],
                      key=lambda x: x["heavy_per_hour"])  # fmt: skip
        ax.plot([x["heavy_per_hour"] for x in rows], [x["battery_pct_per_hour"] for x in rows], "o", color=col,
                label=f"{accel.upper()} compute, 0.05 W capture")  # fmt: skip
    ax.axhline(5, color=MUTED, ls="--", lw=1)
    ax.text(ax.get_xlim()[0], 5, " 5 %/h budget", color=INK2, fontsize=8, va="bottom")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("heavy-perception invocations per hour (measured offline)")
    ax.set_ylabel("battery %/h (ESTIMATED, analytic model)")
    ax.legend(fontsize=7)
    _save(fig, "m8/figs/energy_estimated.png")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--allow-missing", action="store_true")
    a = ap.parse_args()
    for f in (fig_m1, fig_m2, fig_m3, fig_m4, fig_m8):
        try:
            f()
        except Exception as e:
            if not a.allow_missing:
                raise
            print("skip", f.__name__, e)
    dst = REPO / "paper" / "figs"
    dst.mkdir(parents=True, exist_ok=True)
    for p in OUT:
        shutil.copy(p, dst / p.name)
    print("figures:", [str(p.relative_to(REPO)) for p in OUT])


if __name__ == "__main__":
    main()
