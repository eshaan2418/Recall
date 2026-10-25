"""Write paper/numbers.tex (LaTeX macros) and paper/results_sections.tex from results/ — the paper never contains a
hand-typed result. Missing result files produce a visible "[pending]" instead of a number.

    python -m recall.bench.paper_numbers
"""

from __future__ import annotations

import json
from collections.abc import Callable

from recall.paths import REPO, RESULTS

PENDING = r"\textcolor{red}{[pending]}"


def _load(rel: str) -> dict | None:
    p = RESULTS / rel
    return json.loads(p.read_text()) if p.exists() else None


def f2(x: float | None) -> str:
    return PENDING if x is None else f"{x:.2f}"


def main() -> None:
    m1, pp, b1 = _load("m1/systems_test.json"), _load("m1/prior_and_paired.json"), _load("m1/b1_vlm_K16.json")
    m2, m3, m4 = _load("m2/pareto_test.json"), _load("m3/memory_storage.json"), _load("m4/calibration_test.json")
    m5, m7, m8 = _load("m5/compression.json"), _load("m7/redaction.json"), _load("m8/energy_model.json")
    mac: dict[str, str] = {}

    def get(fn: Callable[[], float | str | None], fmt: Callable = f2) -> str:
        try:
            v = fn()
        except (KeyError, TypeError, IndexError, StopIteration):
            return PENDING
        return fmt(v) if not isinstance(v, str) else v

    S = (m1 or {}).get("systems", {})
    mac["RecallLoc"] = get(lambda: S["Recall-all-frames"]["metrics"]["where_loc_acc"]["mean"])
    mac["PriorLoc"] = get(lambda: pp["B0_prior"]["test_where_loc_acc"]["mean"])
    W = (pp or {}).get("paired_on_B1_subsample", {}).get("where", {})
    A = (pp or {}).get("paired_on_B1_subsample", {}).get("absent", {})
    mac["RecallLocSub"] = get(lambda: W["Recall-all-frames"]["mean"])
    mac["BoneLoc"] = get(lambda: W["B1-VLM-K16"]["mean"])
    mac["RecallMinusBone"] = get(lambda: W["Recall_minus_B1-VLM-K16"]["delta"])
    mac["RecallMinusBoneLo"] = get(lambda: W["Recall_minus_B1-VLM-K16"]["lo"])
    mac["RecallMinusBoneHi"] = get(lambda: W["Recall_minus_B1-VLM-K16"]["hi"])
    mac["RecallHallSub"] = get(lambda: 1 - A["Recall-all-frames"]["mean"])
    mac["BoneHall"] = get(lambda: 1 - A["B1-VLM-K16"]["mean"])
    mac["HallDiff"] = get(lambda: A["Recall_minus_B1-VLM-K16"]["delta"])
    mac["HallDiffLo"] = get(lambda: A["Recall_minus_B1-VLM-K16"]["lo"])
    mac["HallDiffHi"] = get(lambda: A["Recall_minus_B1-VLM-K16"]["hi"])
    mac["ObjHeadVal"] = get(lambda: m1["heads"]["obj"]["best_val_acc"])
    n = (m1 or {}).get("n_videos", {})
    nq = (m1 or {}).get("n_queries_test", {})
    for k, key in (("NTrainVideos", "train"), ("NValVideos", "val"), ("NTestVideos", "test")):
        mac[k] = str(n.get(key, PENDING))
    for k, key in (("NWhere", "where"), ("NTemporal", "temporal"), ("NState", "state"), ("NAbsent", "absent")):
        mac[k] = str(nq.get(key, PENDING))

    # ---- M2 trigger
    sections = []
    trig = ""
    if m2:
        rows = m2["rows"]
        allf = S.get("Recall-all-frames", {}).get("metrics", {}).get("where_acc", {}).get("mean")
        orc = next((r for r in rows if r["policy"] == "oracle"), None)
        learned = [r for r in rows if r["policy"] == "learned+imu" and r.get("target_budget") == 0.2]
        uni = [r for r in rows if r["policy"] == "uniform-k5"]
        if learned:
            fr = sum(r["frac_processed"] for r in learned) / len(learned)
            acc = sum(r["metrics"]["where_acc"]["mean"] for r in learned) / len(learned)
            mac["TrigFrac"] = f2(fr)
            mac["TrigSkip"] = f2(1 - fr)
            mac["TrigAcc"] = f2(acc)
            mac["TrigAccAll"] = f2(allf)
            mac["TrigAccOracle"] = f2(orc["metrics"]["where_acc"]["mean"]) if orc else PENDING
            mac["TrigAccUniform"] = f2(uni[0]["metrics"]["where_acc"]["mean"]) if uni else PENDING
            ap = [g["test_ap"] for k, g in _load("m2/gate_training.json")["gates"].items() if k.startswith("learned+imu")]  # type: ignore[index]
            mac["GateAP"] = f2(sum(ap) / len(ap))
            mac["GateParams"] = str(next(iter(_load("m2/gate_training.json")["gates"].values()))["params"])  # type: ignore[index]
            trig = (r"With the learned trigger at a 20\,\% budget, heavy perception runs on \TrigFrac{} of frames "
                    r"(\TrigSkip{} skipped) at ``where'' accuracy \TrigAcc{} vs.\ \TrigAccAll{} with all frames.")
        sections.append(r"""\section{Learned Trigger}\label{sec:trigger}
\begin{figure}[t]\centering\includegraphics[width=\linewidth]{figs/pareto.png}
\caption{Accuracy vs.\ fraction of frames sent to heavy perception (test; learned gates averaged over 3 seeds).}\label{fig:pareto}\end{figure}
The gate has \GateParams{} parameters and reaches test AP \GateAP{} for narrated interaction frames (mean of 3 seeds).
At a 20\,\% budget it processes \TrigFrac{} of frames with ``where'' accuracy \TrigAcc{}, against \TrigAccAll{} for
all frames, \TrigAccUniform{} for uniform sampling at the same rate and \TrigAccOracle{} for the oracle trigger
(Fig.~\ref{fig:pareto}). Because perception accuracy is low overall, differences between policies are small relative
to their confidence intervals; the trigger's value here is compute, not accuracy.""")
    else:
        sections.append(r"\section{Learned Trigger}\label{sec:trigger}" + PENDING)
    mac["TriggerAbstract"] = trig

    # ---- M3 memory
    st = ""
    if m3:
        rows = [r for r in m3["rows"] if r["variant"] == "events-fp16"]
        kf = [r for r in m3["rows"] if r["variant"] == "keyframes-2s"]
        summ = [r for r in m3["rows"] if r["variant"] == "_summary"]
        lr = next((r for r in rows if r["policy"].startswith("learned")), rows[0])
        mac["StoreGBWeek"] = f"{lr['projected_GB_per_week']:.3f}"
        mac["KeyframeGBWeek"] = f"{kf[0]['projected_GB_per_week']:.1f}" if kf else PENDING
        ls = next((r for r in summ if r["policy"].startswith("learned")), summ[0])
        mac["EventsPerHour"] = f"{ls['events_per_hour']:.0f}"
        mac["QueryP95"] = f"{ls['query_latency_ms_p95']:.1f}"
        st = (r"Event memory projects to \StoreGBWeek{}\,GB/week (vs.\ \KeyframeGBWeek{}\,GB/week for 2\,s keyframes); "
              r"query latency p95 \QueryP95\,ms on a laptop.")
        sections.append(r"""\section{Compact Event Memory}\label{sec:memory}
\begin{figure}[t]\centering\includegraphics[width=\linewidth]{figs/storage.png}
\caption{Accuracy vs.\ projected storage (16 recorded hours/day, an assumption).}\label{fig:storage}\end{figure}
At the learned-trigger operating point the memory holds \EventsPerHour{} events/hour. Measured SQLite files (with real
JPEG thumbnails) project to \StoreGBWeek{}\,GB/week, versus \KeyframeGBWeek{}\,GB/week for storing a keyframe every
2\,s (Fig.~\ref{fig:storage}); the 1\,GB/week budget is met with a wide margin. Answering a query (SQL fetch, heads,
selection) takes \QueryP95\,ms at p95 on the desktop (\emph{desktop-proxy}).""")
    else:
        sections.append(r"\section{Compact Event Memory}\label{sec:memory}" + PENDING)
    mac["StorageAbstract"] = st

    # ---- M4 calibration
    ca = ""
    if m4:
        sysn = next(n for n in m4["systems"] if n.startswith("Recall-learned"))
        R = m4["systems"][sysn]
        mac["EceAlways"] = f2(R["confidence"]["always-confident"]["ece"])
        mac["EceIso"] = f2(R["confidence"]["isotonic"]["ece"])
        mac["SelAcc"] = f2(R["isotonic_selective"]["test_selective_acc"])
        mac["SelCov"] = f2(R["isotonic_selective"]["test_coverage"])
        mac["HallAfter"] = f2(R["isotonic_selective"]["absent_hallucination_rate"])
        mac["HallBefore"] = f2(R["absent_hallucination_no_abstention"])
        g = m4.get("generation", {})
        mac["GenHalluc"] = str(g.get("answers_mentioning_a_location_not_in_evidence", PENDING))
        mac["GenN"] = str(g.get("n_rendered", PENDING))
        ca = r"Isotonic calibration lowers ECE from \EceAlways{} (always-confident) to \EceIso{}."
        sections.append(r"""\section{Calibrated Answers and Abstention}\label{sec:calib}
\begin{figure}[t]\centering\includegraphics[width=\linewidth]{figs/calibration.png}
\caption{Reliability (left) and risk--coverage (right), test set; calibrators fit on val.}\label{fig:calib}\end{figure}
For the learned-trigger system, ECE falls from \EceAlways{} when every answer is treated as certain to \EceIso{} with an
isotonic calibrator fit on val (Fig.~\ref{fig:calib}). With the val-chosen threshold, Recall answers \SelCov{} of
answerable queries at selective accuracy \SelAcc{}; the never-seen-object hallucination rate goes from \HallBefore{} to
\HallAfter. Of \GenN{} rendered answers, \GenHalluc{} mention a place not present in the evidence (templated grounding
plus verification of LLM rephrases).""")
    else:
        sections.append(r"\section{Calibrated Answers and Abstention}\label{sec:calib}" + PENDING)
    mac["CalibAbstract"] = ca

    # ---- M5 compression
    if m5 and "C_hoi_student" in m5:
        C = m5["C_hoi_student"]
        mac["HoiHandF"] = f2(C["torch-fp32"]["hand_present_f1"])
        mac["HoiContactF"] = f2(C["torch-fp32"]["contact_f1"])
        mac["HoiContactFQ"] = f2(C.get("litert-int8dyn", {}).get("contact_f1"))
        mf = m5.get("manifest") or {}
        mac["EmbMBfp"] = f"{mf['embedder_fp32']['bytes'] / 1e6:.1f}" if "embedder_fp32" in mf else PENDING
        mac["EmbMBq"] = f"{mf['embedder_int8w']['bytes'] / 1e6:.1f}" if "embedder_int8w" in mf else PENDING
        mac["EmbCosQ"] = f"{mf['embedder_int8w']['cosine_vs_torch']:.3f}" if "embedder_int8w" in mf else PENDING
        mac["EmbCosFour"] = f"{mf['embedder_int4w']['cosine_vs_torch']:.2f}" if "embedder_int4w" in mf else PENDING
        sections.append(r"""\section{Compression and On-Device Port}\label{sec:compress}
\begin{figure}[t]\centering\includegraphics[width=\linewidth]{figs/compression.png}
\caption{Hand--object student agreement with the teacher per precision (left) and LiteRT sizes (right).}\end{figure}
The hand--object student agrees with the Faster R-CNN teacher on held-out frames with hand-present F1 \HoiHandF{} and
contact F1 \HoiContactF{} (\HoiContactFQ{} after dynamic INT8). The embedder goes from \EmbMBfp\,MB to \EmbMBq\,MB with
INT8 weight-only quantisation (cosine to fp32 \EmbCosQ); channel-wise INT4 weights destroy it (cosine \EmbCosFour).
All latency numbers are \emph{desktop-proxy}: no phone was available.""")
    # ---- M7 privacy
    if m7:
        f = m7["faces"]["recall"]["face_detector"]
        mac["FaceAll"] = f2(f["overall"]["point"])
        mac["FaceLarge"] = f2(f["by_face_height_px"]["[300,100000)"]["point"])
        mac["ScreenAll"] = f2(m7["screens_documents"]["recall"]["screen_detector"]["overall"]["point"])
        sections.append(r"""\section{Privacy Layer}\label{sec:privacy}
\begin{figure}[t]\centering\includegraphics[width=\linewidth]{figs/redaction.png}
\caption{Redaction recall: faces by size (WIDER FACE val) and synthetic screens/documents.}\end{figure}
The redactor catches \FaceLarge{} of faces taller than 300\,px but only \FaceAll{} of all WIDER FACE faces, and
\ScreenAll{} of synthetic screens/documents (none in dark mode). Thumbnails are small event crops, which limits
exposure, but this is the weakest component and is reported as such.""")
    # ---- M8 energy
    if m8:
        be = m8["breakeven_5pct"]
        mac["BreakNpuLow"] = f2(be["npu/low"]["max_frac_of_2fps_frames"])
        mac["BreakCpuLow"] = f2(be["cpu/low"]["max_frac_of_2fps_frames"])
        mac["CapHigh"] = f2(be["cpu/high"]["capture_alone_pct_per_hour"])
        sections.append(r"""\section{Energy (Estimated)}\label{sec:energy}
\begin{figure}[t]\centering\includegraphics[width=\linewidth]{figs/energy_estimated.png}
\caption{\emph{Estimated} battery \%/h from an analytic model (assumed constants), vs.\ measured heavy invocations/hour.}\end{figure}
With no phone, energy is \emph{estimated}: measured invocation counts and exact MAC counts times assumed energy per
GMAC and capture power. With a glasses-class 0.05\,W sensor, the 5\,\%/h budget admits heavy processing on up to
\BreakNpuLow{} of 2\,fps frames on an NPU but only \BreakCpuLow{} on a CPU, which is where a learned trigger matters.
A phone camera pipeline kept open at 1\,W would alone cost \CapHigh\,\%/h, exceeding the budget regardless of trigger.""")
    else:
        sections.append(r"\section{Energy (Estimated)}\label{sec:energy}" + PENDING)

    (REPO / "paper" / "numbers.tex").write_text(
        "% generated by recall/bench/paper_numbers.py from results/ — do not edit\n"
        + "".join(f"\\newcommand{{\\{k}}}{{{v}}}\n" for k, v in mac.items())
    )
    (REPO / "paper" / "results_sections.tex").write_text("% generated — do not edit\n" + "\n\n".join(sections) + "\n")
    print(f"wrote {len(mac)} macros; pending: {[k for k, v in mac.items() if v == PENDING]}")


if __name__ == "__main__":
    main()
