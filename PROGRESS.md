# PROGRESS

Status legend: `todo` · `in-progress` · `done` · `blocked-with-fallback` · `awaiting-human`

| Milestone | Status | Evidence |
|---|---|---|
| M0 Bootstrap | done | ENVIRONMENT.md, `make test` green (synthetic fixtures), CI workflow, Dockerfile, downloader |
| M1 Offline pipeline + baselines | in-progress | runner `recall/bench/run_m1.py`; B1 `recall/baselines/vlm.py` running |
| M2 Learned trigger | in-progress | runner `recall/bench/run_m2.py` (waits on M1 + full feature cache) |
| M3 Compact memory | in-progress | runner `recall/bench/run_m3.py` |
| M4 Calibration & abstention | in-progress | runner `recall/bench/run_m4.py` |
| M5 Compression & port | in-progress | `recall/compress/{export,hoi_student}.py`, `recall/bench/run_m5.py`; export verified on a dummy gate |
| M6 Android app | in-progress | `android/` (agent building) |
| M7 Privacy layer | in-progress | `recall/privacy/`, `privacy/THREAT_MODEL.md` (agent) |
| M8 Energy/thermal/storage | in-progress (fallback: estimated) | `recall/energy/model.py` (analytic, `estimated`), `recall/energy/harness.py` (on-device, needs phone) |
| M9 Personal-data eval | in-progress (fallback: stand-in) | `recall/data/annotate/server.py`, `recall/data/loaders/personal.py`, `recall/bench/run_m9.py --standin 3` |
| M10 User-study package | awaiting-human | `study/` complete; analysis validated on simulated data (`study/sim_output/`, labelled SIMULATED); `tests/test_study.py` |
| M11 Integration & write-up | todo | `scripts/reproduce_{quick,full}.sh`, `recall/bench/{figures,check_numbers}.py` written |

## Experiment order (once the feature cache has all 58 videos)
1. `run_m1` → heads, val-tuned configs, test metrics for Recall / B2 / B3 / B4 (B1 runs in parallel).
2. `run_m2` (3 seeds × img / img+IMU gates) → Pareto.
3. `run_m3`, `run_m4`.
4. `hoi_student` (train split cache) → `export` (in `.venv-export`) → `run_m5 --parts abcde`.
5. `energy.model`, `run_m9 --standin 3`, `figures`, then paper / README / FINAL_REVIEW, `check_numbers`, `scan`.

## Next action
Wait for the remaining train downloads (P28_*) + extraction, then run step 1.
