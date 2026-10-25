# M10 data schemas

Generated from `recall/study/schema.py` (the source of truth; the analysis validates every file against it and
refuses to run on invalid data). Header-only templates are in `study/templates/`; copy them to `study/data/`
(git-ignored) at the start of the study.

General rules

- UTF-8 CSV, comma-separated, header row exactly as below, one file per table.
- Leave a cell **empty** for NA (e.g. `assistant_correct` in the `none` condition; SUS / trust / usefulness for
  the `none` block). "only for recall/b2" means: must be empty for other conditions and filled for these.
- `participant_id` is the pseudonym `P01`…`P99` only. **No names, initials, emails, dates of birth, free-text
  comments or photos** in any CSV. Free-text debrief notes are not collected (see `data_handling.md`).
- `item` / `hiding_place` / `trial_index` / `block_index` / `condition` are copied from
  `study/assignments/run_sheets/Pxx.md`, so the CSV can be checked against the design.
- Time-to-find: seconds, one decimal, from the moment the experimenter finishes naming the item ("Please find
  the keys — go") until the participant **touches** the item. If not found at the cap (`time_cap_s` in
  `study_config.yaml`, default 180 s), enter the cap and `success = 0`.
- NASA-TLX subscales are entered on the 0–100 scale (21 tick marks → multiples of 5). For `tlx_performance`,
  0 = "perfect", 100 = "failure" (the standard TLX polarity — do not reverse it).
- SUS items 1–5 (1 = strongly disagree … 5 = strongly agree), entered as ticked (the analysis does the SUS
  odd/even recoding). Trust and usefulness items 1–7 (1 = strongly disagree … 7 = strongly agree).
- An optional `simulated` column (0/1) is allowed. Real data must not contain `simulated = 1`; the real-data
  analysis aborts if it does.

### `participants.csv`

| column | type | required | range / values | description |
|---|---|---|---|---|
| `participant_id` | str | yes | `^P\d{2}$` | Pseudonymous ID P01..P99 (key in the paper link sheet) |
| `design_row` | int | yes | 0 – 98 | Row of the counterbalancing design (from assignments.csv) |
| `age_band` | cat | yes | 18-24, 25-34, 35-44, 45-54, 55-64, 65+, prefer_not | Coarse age band only |
| `tech_familiarity` | int | no | 1 – 5 | Self-rated smartphone familiarity 1-5 |
| `familiar_with_space` | int | yes | 0 – 1 | 1 if the participant knew the study room beforehand |
| `session_completed` | int | yes | 0 – 1 | 1 if all 3 blocks were completed |

### `trials.csv`

| column | type | required | range / values | description |
|---|---|---|---|---|
| `participant_id` | str | yes | `^P\d{2}$` |  |
| `trial_index` | int | yes | 1 – 99 | 1-based position of the trial in the session |
| `block_index` | int | yes | 1 – 3 | 1-based block (condition) position |
| `condition` | cat | yes | recall, none, b2 |  |
| `item` | str | yes |  | Item name exactly as in study_config.yaml |
| `hiding_place` | str | yes |  | Hiding place name exactly as in study_config.yaml |
| `time_to_find_s` | float | yes | >= 0 | Seconds from item named to item touched; = time cap if not found |
| `success` | int | yes | 0 – 1 | 1 if found within the time cap |
| `n_queries` | int | no | >= 0 | Queries typed/spoken to the assistant (0 for none) |
| `assistant_correct` | int | only for recall/b2 | 0 – 1 | 1 if the assistant's first answer showed the right place; NA for none |
| `assistant_confidence` | float | only for recall | 0 – 1 | Confidence Recall displayed for its first answer (0-1); NA for none/b2 |
| `followed_assistant_first` | int | only for recall/b2 | 0 – 1 | 1 if the participant searched the suggested place first; NA for none |

### `questionnaires.csv`

| column | type | required | range / values | description |
|---|---|---|---|---|
| `participant_id` | str | yes | `^P\d{2}$` |  |
| `condition` | cat | yes | recall, none, b2 |  |
| `block_index` | int | yes | 1 – 3 |  |
| `tlx_mental` | int | yes | 0 – 100 | NASA-TLX mental 0-100 (steps of 5; performance: 0 = perfect) |
| `tlx_physical` | int | yes | 0 – 100 | NASA-TLX physical 0-100 (steps of 5; performance: 0 = perfect) |
| `tlx_temporal` | int | yes | 0 – 100 | NASA-TLX temporal 0-100 (steps of 5; performance: 0 = perfect) |
| `tlx_performance` | int | yes | 0 – 100 | NASA-TLX performance 0-100 (steps of 5; performance: 0 = perfect) |
| `tlx_effort` | int | yes | 0 – 100 | NASA-TLX effort 0-100 (steps of 5; performance: 0 = perfect) |
| `tlx_frustration` | int | yes | 0 – 100 | NASA-TLX frustration 0-100 (steps of 5; performance: 0 = perfect) |
| `sus_1` | int | only for recall/b2 | 1 – 5 | SUS item 1-5 |
| `sus_2` | int | only for recall/b2 | 1 – 5 | SUS item 1-5 |
| `sus_3` | int | only for recall/b2 | 1 – 5 | SUS item 1-5 |
| `sus_4` | int | only for recall/b2 | 1 – 5 | SUS item 1-5 |
| `sus_5` | int | only for recall/b2 | 1 – 5 | SUS item 1-5 |
| `sus_6` | int | only for recall/b2 | 1 – 5 | SUS item 1-5 |
| `sus_7` | int | only for recall/b2 | 1 – 5 | SUS item 1-5 |
| `sus_8` | int | only for recall/b2 | 1 – 5 | SUS item 1-5 |
| `sus_9` | int | only for recall/b2 | 1 – 5 | SUS item 1-5 |
| `sus_10` | int | only for recall/b2 | 1 – 5 | SUS item 1-5 |
| `trust_1` | int | only for recall/b2 | 1 – 7 | Trust Likert 1-7 |
| `trust_2` | int | only for recall/b2 | 1 – 7 | Trust Likert 1-7 |
| `trust_3` | int | only for recall/b2 | 1 – 7 | Trust Likert 1-7 |
| `useful_1` | int | only for recall/b2 | 1 – 7 | Usefulness 1-7 |
| `useful_2` | int | only for recall/b2 | 1 – 7 | Usefulness 1-7 |

