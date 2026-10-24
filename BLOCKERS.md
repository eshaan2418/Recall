# BLOCKERS

| Date | Blocker | Fallback | Impact on claims |
|---|---|---|---|
| 2026-10-24 | No Ego4D credentials (`.env` absent) | EPIC-KITCHENS-100 subset (14.4 h) with narration-derived "Where-is-X", state and temporal query sets. Ego4D loaders + metric code written and unit-tested on fixtures, ready to run once access exists | **No official Ego4D VQ2D/NLQ numbers.** Accuracy claims are about EPIC-derived tasks only; comparability to published Ego4D EM results is not claimed |
| 2026-10-24 | No Android phone connected | Emulator (functional tests) + desktop-proxy latency + analytic energy model | **Battery %/hour and thermal are `estimated`, not measured.** The 5%/h budget question is answered only conditionally on the energy model's assumptions |
| 2026-10-24 | No cloud VLM key | Local open VLM (Qwen2.5-VL) as B1 | B1 is weaker than a frontier cloud model; "beats cloud VLM" cannot be claimed, only "beats a large open VLM given sampled frames" |
| 2026-10-24 | No personal recordings | M9 tooling built and tested on EPIC clips used as stand-in personal data | No real-life (non-kitchen, multi-room) evaluation |
| 2026-10-24 | No CUDA GPU (Apple M5 Pro, MPS) | Train only small models; pretrained teachers | No large-model fine-tuning |
