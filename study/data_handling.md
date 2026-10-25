# M10 data-handling plan

Principles: **data minimisation, nothing leaves the device, pseudonymous IDs, video deleted at session end,
only aggregate results published.**

## 1. What is recorded

| # | data | form | identifiable? | purpose |
|---|---|---|---|---|
| D1 | Chest-camera video of the encoding walk (~15 min) + the Recall / B2 memory index derived from it | Files on the study phone | Potentially (room, hands, voice is NOT recorded — mic off) | Input to the assistants during the session only |
| D2 | Trial measurements (`trials.csv`) | Paper run sheet → CSV | Pseudonymous (Pxx) | Primary / secondary outcomes |
| D3 | Questionnaires (`questionnaires.csv`) | Paper → CSV | Pseudonymous | NASA-TLX, SUS, trust, usefulness |
| D4 | Participant attributes (`participants.csv`) | Paper → CSV | Pseudonymous; age **band** only | Description of the sample |
| D5 | Consent form (name, signature) | Paper | Identifying | Proof of consent |
| D6 | Link sheet (name ↔ Pxx) | Paper | Identifying | Withdrawal requests only |

Not collected: names/initials/contact details in any data file, exact age, gender, audio, photos/videos of
faces, free-text comments, timestamps of sessions in the data files, device identifiers.

## 2. Where it is stored

- D1: the **study phone** only (airplane mode, Wi-Fi + Bluetooth off for the whole session; no cloud backup
  account signed in; no USB transfer). The analysis pipeline never reads D1.
- D2–D4: `study/data/` on the researcher's laptop, **git-ignored** (see `.gitignore`: `study/data/`), on an
  encrypted disk (FileVault). Never committed, emailed, synced to cloud storage, or pasted into an LLM / chat tool.
  One encrypted offline backup (e.g. an encrypted USB drive kept with the paper forms) is allowed.
- D5, D6: paper, in a locked folder/drawer; D6 kept separately from D2-paper (run sheets).
- Outputs: `make study-analysis` writes aggregate results only (`results/m10/report.md`, `analysis.json`:
  medians, model coefficients, test statistics; no per-participant rows). Only these may be committed/published.
- Simulated data (`study/sim_output/`) contain no real data and are clearly labelled SIMULATED.

## 3. Retention and deletion

| data | deleted when | how |
|---|---|---|
| D1 video + index | End of the participant's own session (in front of them); immediately if they stop early | App "Wipe all" (deletes the files and the index); experimenter ticks the time on the run sheet; spot-check the phone's file browser is empty before the next participant |
| Paper run sheets | After transcription is double-checked (≤ 7 days) | Shred |
| D6 link sheet | On the withdrawal deadline stated in the consent form (e.g. 4 weeks after the last session) | Shred → the CSVs become anonymous |
| D2–D4 CSVs | [5 years after publication / per institution policy] | Secure delete of `study/data/` and the backup |
| D5 consent forms | [Per institution policy, e.g. 5 years] | Shred |

Withdrawal before the deadline: look up the code on D6, delete every row with that `participant_id` from all
three CSVs (and the backup), shred the consent form only if the institution allows, note "Pxx withdrawn" in the
experimenter notebook without the name. Re-run `make study-analysis`.

## 4. Pseudonymous IDs

`P01`… assigned in order of participation from the pre-generated run sheets; they carry no information about
the person. The analysis rejects IDs not matching `P\d\d` (so pilot IDs `X01` and any accidental names fail
validation).

## 5. Bystanders

The session runs in a room with no other people; if someone enters during recording, recording stops and that
segment is deleted. No screens, documents or photos with personal information may be in camera view.

## 6. Incidents

If video or a CSV is copied off the device by mistake, delete the copy, record what happened (without names) in
the experimenter notebook and inform [ethics committee / supervisor] as your approval requires.
