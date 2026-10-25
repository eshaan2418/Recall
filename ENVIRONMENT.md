# ENVIRONMENT (probed 2026-10-24, M0)

| Item | Value |
|---|---|
| Host | Apple MacBook, **Apple M5 Pro** (18 CPU cores: 6P + 12E; 20-core GPU, Metal 4) |
| RAM | 48 GB unified |
| Disk | 770 GB free |
| OS | macOS 27.0.1 (Darwin 27.0.0, arm64) |
| GPU / accel | No CUDA. PyTorch MPS backend available (torch 2.14.1). |
| Python | 3.12.14 (mise) in `.venv` via uv 0.12.13; transformers 5.20.1, numpy 2.5.4 |
| Package index | Corporate PyPI/Maven mirror via user-level pip/gradle config (credentials live **outside** the repo, never copied in) |
| Docker | 29.7.2 (daemon running) |
| Tools | ffmpeg 9.0.2, gitleaks 8.30.1, tectonic 0.17.0 (LaTeX), gh, brew |
| Java / Android | No JDK on PATH (brew `openjdk` keg present). No Android SDK/NDK, no `adb` → installed into user dir at M6 (see DECISIONS) |
| Connected device | **None** (`adb` absent; no phone connected) |
| Network | PyPI, Hugging Face, data.bris.ac.uk reachable. data.bris ≈1.2 MB/s per connection; ≈9 MB/s with 8–10 parallel range requests |

## Dataset access status
| Dataset | Status |
|---|---|
| Ego4D (EM: VQ2D, NLQ) | **No access** — no credentials in `.env` (no `.env` exists). Blocked; see BLOCKERS.md |
| EPIC-KITCHENS-100 | **Available** — publicly downloadable (CC BY-NC 4.0). Subset of 58 narrated videos / 14.2 h downloaded locally (`configs/epic_subset.json`) |
| EPIC-100 hand-object bboxes (Shan et al. 2020 detector output) | Available, downloaded per selected video |
| EPIC-100 IMU (accl/gyro) | Available for extension videos (28 of 58 subset videos) |
| Personal recordings (`data/personal/raw/`) | **None present** |
| Cloud VLM key | **None** |

## Fallback-ladder rungs chosen
| Resource | Rung | Consequence |
|---|---|---|
| Training compute | **Fallback 1/2 hybrid**: Apple-silicon GPU (MPS), no CUDA | Train only small models (trigger gate, calibration heads, student detector heads); teachers are pretrained |
| Datasets | **Fallback 1**: EPIC-KITCHENS-100 only (Ego4D blocked) | No official Ego4D VQ2D/NLQ numbers. Replaced with EPIC-derived "Where-is-X", state and temporal query sets with narration-derived ground truth |
| Device | **Fallback 1 → 2**: Android emulator for functional tests; desktop runtime as latency proxy | **No `on-device` energy/thermal claims.** All energy numbers are `estimated` |
| Cloud baseline | **Fallback 1**: open VLM run locally as "large-model baseline" B1 | B1 is not a frontier cloud model; stated in all tables |
