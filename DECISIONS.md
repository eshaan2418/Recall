# DECISIONS

Dated decisions + rationale. Newest at the bottom.

## 2026-10-24 — M0
- **Python package lives in `recall/`** (`recall/perception`, `recall/trigger`, …) rather than top-level dirs, so it's importable and type-checkable. Top-level `android/`, `paper/`, `results/`, `study/` (docs), `data/` (git-ignored raw data) follow the requested layout.
- **Primary dataset = EPIC-KITCHENS-100** (Ego4D unavailable). We use the 456×256 RGB frame tars (smaller than the 1080p MP4s, and the same resolution class a low-power always-on pipeline would see).
- **Participant-disjoint splits**: train {P02,P04,P06,P24,P28}, val {P18,P30}, test {P01,P03,P10,P11,P23,P37}, ≈1.0–1.3 h per participant, videos 6–30 min, chosen by count of narrations with an explicit location ("put X in Y"). The selection is deterministic (`recall/data/loaders/epic.py::select_subset`) and recorded in `configs/epic_subset.json` (59 videos, 14.4 h). Test participants never touch trigger or calibration training.
- **Ground truth from narrations.** EPIC narrations are timestamped verb–noun annotations made by the participants. "Put X in/on Y" gives object, location and time; "take/pick up X" ends a location. This yields the custom "Where is X?" set without new labeling. Limitation: narration-based GT only covers *narrated* interactions; un-narrated placements are invisible to the GT.
- **Base stream rate 2 fps** for offline simulation (the always-on gate sees 2 fps; heavy perception is invoked on a subset). Native EPIC fps is 50–60; 2 fps keeps 14 h tractable (≈104 k frames) and matches always-on budgets reported for wearables.
- **Hand–object interaction (HOI) teacher** = the EPIC-100 released outputs of Shan et al. (CVPR 2020) hand–object detector, run by the dataset authors at full frame rate. We treat it as the heavy model's HOI head and charge its cost (Faster R-CNN R101) when used. On-device we replace it with a distilled student (M5).
- **Large-model baseline B1** = an open VLM run locally (Qwen2.5-VL), because no cloud API key exists. Labeled "large open VLM", not "cloud frontier model", everywhere.
- **No phone** → emulator for functional Android tests; all energy numbers are `estimated` from an explicit, documented energy model, never presented as measurements.
- Never copy the user-level pip/gradle mirror credentials into the repo; CI uses public PyPI/Maven.
- **Embedder = MobileCLIP2-S2 (dfndr2b)**, chosen on train+val videos only (`results/m1/select_embedder.json`; 222 object crops / 186 location crops from P02_129, P02_123, P18_*). Zero-shot object top-1 0.149 vs 0.171 for ViT-L-14 (8.5× larger image encoder), location top-1 0.226 vs 0.215. MobileCLIP2-S2 is the only candidate both near-best and plausibly phone-deployable (≈36 M image-encoder params). ViT-L-14 is kept as the "teacher" reference in M5.
- Derived query files (`configs/queries/*.jsonl`) are committed: they contain EPIC narration strings (CC BY-NC 4.0, attribution: Damen et al., "Rescaling Egocentric Vision", IJCV 2022) but no frames.
