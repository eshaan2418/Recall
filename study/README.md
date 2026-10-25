# M10 — User-study package

A within-subjects study (planned n = 12; 10–15 acceptable) comparing three ways of finding objects a participant
put away ~30 minutes earlier while wearing a chest camera:

| condition | what the participant gets |
|---|---|
| `recall` | Recall: place answer + calibrated confidence + evidence thumbnail (may abstain) |
| `b2` | Baseline B2: frame sampling + CLIP retrieval → the single best-matching frame, no confidence |
| `none` | No assistant (memory only) |

**Outcomes.** Primary: time-to-find (s, capped at 180 s). Secondary: success (found within the cap), raw NASA-TLX
(all conditions), SUS, trust and usefulness Likert (assistant conditions). Descriptive: assistant accuracy and
appropriate reliance (followed the assistant when it was right vs wrong; Recall confidence AUROC).

**Pre-specified hypotheses.** H1 Recall is faster than B2; H2 Recall is faster than no assistant (primary family,
Holm). Secondary family (Holm): success Recall vs B2 / vs none; TLX pairwise; SUS, trust, usefulness Recall vs B2.

## Files

| file | what |
|---|---|
| `consent_form.md` | Consent form (fill bracketed fields; or port into your institution's template) |
| `privacy_notice.md` | Participant-facing privacy notice |
| `data_handling.md` | What is recorded, where, retention, deletion, withdrawal; nothing leaves the device |
| `run_session.md` | Experimenter script, verbatim instructions, per-trial recording rules, pilot |
| `questionnaires.md` | NASA-TLX, SUS, trust (3) and usefulness (2) items, mapped to CSV columns |
| `data_schemas.md` | CSV schemas (generated from `recall/study/schema.py`); templates in `templates/` |
| `study_config.yaml` | Items, hiding places, filler tasks, seed, time cap, delay — edit for your site first |
| `assignments/` | Generated: `assignments.csv`, `balance.txt`, `run_sheets/Pxx.md` (one per participant) |
| `sim_output/` | **SIMULATED** data, analysis of simulated data, and simulation-based power. Not results. |
| `data/` | Real CSVs (git-ignored; never commit) |

Code: `recall/study/{design,schema,simulate,analysis,power}.py`; tests: `tests/test_study.py`.

## Design and counterbalancing

- Condition order: Williams design (all 6 orders of 3 conditions → each condition appears equally often in each
  position and each condition immediately follows each other equally often). Balanced in every block of 6
  participants, so n = 6, 12, 18 are exactly balanced; at n = 15 the last 3 rows are an incomplete block.
- Items: 9 items in 3 fixed sets of 3; the set ↔ condition mapping is rotated by a second Latin square (each set
  paired with each condition equally often per 3 participants, shifted each block of 6 so it is not confounded
  with order). Hiding places, put-down order and within-block retrieval order are random per participant from
  the seed in `study_config.yaml`.
- Participant-to-row allocation is a seeded permutation within each block of 6. A replacement for a withdrawn
  participant reuses the same run sheet.

## Commands

```bash
# 1. edit study/study_config.yaml for your room, then (re)generate run sheets — BEFORE P01 only
PYTHONPATH=. .venv/bin/python -m recall.study.design
# 2. validate the pipeline on SIMULATED participants (writes study/sim_output/{data,analysis}/)
PYTHONPATH=. .venv/bin/python -m recall.study.simulate && PYTHONPATH=. .venv/bin/python -m recall.study.analysis --simulated
# 3. simulation-based power / sensitivity (~3 min on 8 cores; writes study/sim_output/power.{md,json})
PYTHONPATH=. .venv/bin/python -m recall.study.power --reps 300 --n 10 12 15
# 4. real analysis: reads study/data/*.csv if present (else prints a notice, exit 0) -> results/m10/
PYTHONPATH=. .venv/bin/python -m recall.study.analysis          # == make study-analysis
```

The real-data analysis validates every CSV against the schema, excludes participants with
`session_completed = 0`, refuses data marked `simulated = 1`, and writes aggregate statistics only. The simulated
analysis refuses to write under `results/`.

## Analysis (pre-specified; `recall/study/analysis.py`)

- **Time (primary):** `log(time) ~ condition + trial_c + (1|participant) + (1|item)`, crossed random intercepts,
  REML (statsmodels MixedLM). Timeouts enter at the cap. Reported as time ratios with 95 % CIs. Sensitivity:
  found-only trials. Small-sample robustness check: paired t-test on per-participant mean log time.
  Fallback if the crossed model fails: `(1|participant)` + item as a fixed effect.
- **Success:** GEE (binomial, exchangeable, clustered by participant); if any condition is at 0 % or 100 %
  (separation, likely with a 180 s cap) → Wilcoxon on per-participant success rates.
- **NASA-TLX:** Friedman omnibus, pairwise Wilcoxon signed-rank. **SUS / trust / usefulness:** Wilcoxon signed-rank
  Recall vs B2. Effect size: matched-pairs rank-biserial r. Holm within each family.

## Power / sensitivity (SIMULATION-BASED — not a result)

From `study/sim_output/power.md` (300 simulated studies per cell, α = .05 two-sided). Assumed variance components
(log-seconds): participant SD 0.35, item SD 0.25, residual SD 0.55; SUS between-person SD 10, within-person noise
SD 6. These are guesses; update them from the pilot and re-run.

- **n = 12, time-to-find, Recall vs B2:** ~80 % power to detect a time ratio of about **0.65–0.70** (Recall
  taking ≤ 65–70 % of B2's time): LMM 77 % at 0.70, 91 % at 0.65; paired t 79 % at 0.65. A 20 % speed-up (ratio
  0.80) has only ~48 % power (LMM). False-positive rate at ratio 1.0 ≈ 0.04–0.05.
- **n = 12, SUS, Recall vs B2:** ~89 % power for +12 SUS points, 61 % for +8.
- n = 15 raises LMM power at ratio 0.70 from 77 % to 90 % (SUS +8: 65 %); n = 10 drops it to 67 % (SUS +12: 77 %).

So the study can detect large practical effects only; smaller differences should be reported as estimates with
CIs, not as "no difference".

## Ethics approval or informal consent

The study involves human participants, video recording in a private-ish space, and minimal risk. Pick one route
**before recruiting anyone**:

1. **Institutional route (required if you are at a university/company with an IRB/REC, or plan to publish in a
   venue that requires it — most HCI venues do).** Submit to your ethics committee: this README (protocol),
   `run_session.md` (procedure), `consent_form.md` + `privacy_notice.md` (port into their template),
   `data_handling.md` (data-management plan) and `questionnaires.md` (instruments). Typical category: minimal-risk /
   expedited / exempt. Note in the application: adults only; no deception beyond neutral assistant labels (revealed
   at debrief); video stays on an offline device and is deleted at session end; pseudonymous data; withdrawal
   until link-sheet destruction. Put the approval reference in the consent form header and keep the letter with
   the consent forms. Do not start until you have it in writing.
2. **Informal route (independent researcher, no institution, results not submitted to a venue requiring IRB).**
   Still obtain **informed written consent** with `consent_form.md` (header: "informal study — no institutional
   review"), give the privacy notice, follow `data_handling.md` exactly, recruit only adults who are not in a
   dependent relationship with you (no direct reports, no paid staff), and do not use the data for anything
   beyond this study. If you later want to publish, check the venue's policy first — retrospective approval is
   often impossible. If you are in the UK/EU, the GDPR still applies to the coded data while the link sheet
   exists.

Either way: run 1–2 pilots (`X01`, `X02`, not analysed) first, then re-run the power script with pilot variances.

## Requirements on the app (for the M6 owner)

The session script assumes the study build of the Android app has: (a) a hidden experimenter setting to switch
between Recall mode and B2 mode (same shell; B2 shows only the best-matching frame, no confidence, no text
answer); (b) video recording with the microphone off; (c) a "Wipe all" action that deletes the recording and the
memory index; (d) works fully offline (airplane mode). If any is missing, run the assistants on the study laptop
offline instead and adjust `run_session.md` §0/§2/§5 accordingly.

## Text for `HUMAN_TODO.md`

```markdown
- **User study (M10)** — unlocks real user-study results (`results/m10/`). Everything else is prepared in `study/`.
  1. Choose the ethics route in `study/README.md` ("Ethics approval or informal consent"). Institutional: submit
     `study/README.md`, `run_session.md`, `consent_form.md`, `privacy_notice.md`, `data_handling.md`,
     `questionnaires.md`; wait for written approval and put the reference in the consent form. Informal: fill in
     the bracketed fields of the consent form and privacy notice; written consent from every participant.
  2. Edit `study/study_config.yaml` (items, hiding places in your room), then `make study-design` to regenerate
     `study/assignments/run_sheets/`. Do this only before the first participant.
  3. Check the study build of the app has B2 mode, mic-off recording, offline operation and "Wipe all"
     (see `study/README.md` → "Requirements on the app").
  4. Run 1–2 pilots (IDs X01/X02), adjust the time cap if > 20 % of no-assistant trials time out, re-run
     `make study-power` with pilot variances.
  5. Recruit 12 adults (P01–P12; P13–P18 are reserve rows). Run each session with `study/run_session.md`
     (~90 min each). Transcribe into `study/data/{participants,trials,questionnaires}.csv` (templates in
     `study/templates/`; git-ignored; never commit or upload).
  6. Run `make study-analysis` → `results/m10/report.md`. Destroy the link sheet on the withdrawal deadline.
```
