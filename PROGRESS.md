# PROGRESS

Status legend: `todo` · `in-progress` · `done` · `blocked-with-fallback` · `awaiting-human`

| Milestone | Status | Evidence |
|---|---|---|
| M0 Bootstrap | done | ENVIRONMENT.md, `make test` green (synthetic fixtures), CI workflow, Dockerfile, downloader |
| M1 Offline pipeline + baselines | in-progress | |
| M2 Learned trigger | todo | |
| M3 Compact memory | todo | |
| M4 Calibration & abstention | todo | |
| M5 Compression & port | todo | |
| M6 Android app | todo | |
| M7 Privacy layer | todo | |
| M8 Energy/thermal/storage | todo | |
| M9 Personal-data eval | todo | |
| M10 User-study package | todo | |
| M11 Integration & write-up | todo | |

## Current plan (M1)
1. Download the EPIC subset (59 videos, 14.4 h); `scripts/download_epic.sh` (resumable).
2. Extract the 2 fps feature cache (`scripts/extract_all.sh`).
3. Train heads (object/location, region + frame) on train, tune on val → `results/m1/heads.json`.
4. Tune answer configs per system on val; evaluate on test: Recall(all frames), B2 frame+CLIP at N ∈ {1,2,5,10,30} s,
   B3 manual tagging q ∈ {0.1,0.25,0.5,1}, B4 oracle trigger; B1 large open VLM on a fixed test subsample.
5. Accept: where/temporal/state/absent metrics with 95% cluster-bootstrap CIs in `results/m1/`.

## Next action
Write `recall/bench/run_m1.py` (heads → val tuning → test systems) while features extract.
