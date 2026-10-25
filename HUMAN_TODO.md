# HUMAN_TODO — things only you can do

Each item says what it unlocks. Nothing here blocks the rest of the build; all affected milestones run a fallback.

1. **Ego4D access** (unlocks official VQ2D / NLQ numbers in M1).
   - Sign the license at https://ego4d-data.org/docs/start-here/ ; wait for AWS credentials.
   - Create `.env` at repo root (git-ignored) with `AWS_ACCESS_KEY_ID=...` and `AWS_SECRET_ACCESS_KEY=...`.
   - Then run `make data-ego4d && make eval-ego4d`.
2. **Android phone** (unlocks `on-device` latency / battery / thermal in M5, M8).
   - Enable Developer options → USB debugging; connect; accept the RSA prompt; check `adb devices` lists it.
   - Then run `make android-install && make energy` (takes ~5 × 60 min; keep the phone unplugged from power except via a USB power-blocker if you have one — see `recall/energy/README.md`).
3. **Personal recordings** (unlocks M9).
   - Record 2–5 h of chest-mounted first-person video of your own routines; only yourself or consenting people.
   - Drop the files in `data/personal/raw/` (git-ignored). Then run `make annotate-personal` to label put-down events and `make eval-personal`.
4. **Cloud VLM baseline (optional)**: put `GEMINI_API_KEY=...` in `.env` and run `make baseline-cloud ALLOW_CLOUD=1`. It only ever sends public EPIC frames, or personal data if you also pass `--allow-cloud` yourself.
5. **User study (M10)**: get ethics/IRB approval or informal written consent (see `study/README.md`), run the sessions with `study/run_session.md`, put the CSVs in `study/data/`, and run `make study-analysis`.
6. **Commit signing**: commits are signed through your 1Password SSH agent. When several sessions commit at once, signing requests queue until you approve them in 1Password; commits in this repo may sit pending until then. Nothing was changed in your git config.
