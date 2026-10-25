# M10 session script (experimenter)

One session ≈ 90 min, one participant, one experimenter. Read the **bold-quoted** lines verbatim; everything
else is an instruction to you. Do not start a session without a signed consent form (`consent_form.md`) and
ethics approval or the informal-consent route agreed in `study/README.md`.

Conditions (within-subjects, every participant does all three; order from the run sheet):

| code | participant-facing name | what the participant gets |
|---|---|---|
| `recall` | "Assistant 1" or "2" (order of first use) | Recall: typed/spoken question → place answer + confidence + evidence thumbnail; may say it is unsure |
| `b2` | the other number | Baseline B2: frame sampling + CLIP retrieval → the single best-matching frame, no confidence, no text answer |
| `none` | "on your own" | No assistant; memory only |

Both assistants run in the **same app shell on the same study phone**, with the mode set by the experimenter in
the hidden settings screen, so they differ only in what is shown. Never call either one "Recall", "baseline",
"new" or "old" in front of the participant.

---

## 0. Before the participant arrives (15 min)

1. Open `study/assignments/run_sheets/Pxx.md` for the next free participant ID (in order P01, P02, …). If a
   participant withdraws or does not finish, the replacement takes the **same** run sheet (same design row).
2. Check the room: no other people, no photos of people, no documents or screens with personal information in
   camera view. All hiding places from `study_config.yaml` exist and are empty. The 9 items are on the start tray.
3. Study phone: airplane mode ON, Wi-Fi OFF, Bluetooth OFF (nothing leaves the device). Storage has room for
   30 min of video. Previous participant's video and memory index are deleted (Settings → Study → "Wipe all").
   Battery ≥ 60 %.
4. Chest mount + phone ready; stopwatch; printed questionnaires (`questionnaires.md`) × 3; pen; run sheet.

## 1. Welcome and consent (10 min)

**"Thanks for coming. Today you'll put some everyday objects away around this room while wearing a small
camera, then later try to find them again — sometimes on your own and sometimes with one of two phone
assistants. We're testing the assistants, not you. The whole session takes about an hour and a half."**

- Give the privacy notice (`privacy_notice.md`) and consent form (`consent_form.md`). Let them read; answer
  questions; both sign. Write the participant ID (from the run sheet) on the **link sheet** only — the link sheet
  is the only place where the name and ID appear together; it stays in the locked folder.
- Record `participants.csv` row: `participant_id`, `design_row` (top of run sheet), `age_band`,
  `tech_familiarity` (ask: "On a scale of 1 to 5, how comfortable are you with smartphone apps?"),
  `familiar_with_space` (1 if they have been in this room before), `session_completed` (fill in at the end).

## 2. Encoding walk (10–15 min, camera ON)

**"I'll now start the camera on your chest. I'll hand you cards one at a time. Some ask you to put an object
somewhere, some ask you to do a small task. Please do them at a normal pace, as if you were tidying up at home.
You don't need to memorise anything specially. Please don't talk about where things are."**

1. Mount the phone, start recording in the Recall app ("Record" button). Say the time aloud for the log.
2. Hand cards in the run-sheet order (Phase 1). Check that each put-down happens in front of the camera (the
   participant's hands should be visible). If the participant blocks the camera or puts an item in the wrong
   place, note the actual place on the run sheet (the actual place is the ground truth) — do not correct them.
3. If anyone else enters the room, stop recording; the recording for that interval is deleted.
4. Stop recording. The app indexes the video on-device during the delay (both Recall and B2 use the same
   recording). Do not let the participant see the screen.

## 3. Retention delay (≥ 30 min, camera OFF)

- Leave the room together. Filler (not analysed): a 10-min unrelated paper puzzle, a drink, a break.
- **Practice (5 min):** on the same phone, using the experimenter's own pre-recorded demo clip (not the
  participant's video), show each assistant mode once: **"This is how you'd ask: type or say 'where is the red
  mug?'. Assistant 1 shows you …; Assistant 2 shows you …"**. One practice question per assistant. Do **not**
  practise in the study room.
- Check the run sheet: the first retrieval trial must start ≥ `retention_delay_min` after the last put-down.

## 4. Retrieval blocks (3 blocks × ~10 min)

Follow the block order on the run sheet. Before each block, set the phone mode (hidden settings) and say:

- `none`: **"For the next three objects, please find them on your own, without the phone."** (Take the phone.)
- `recall` / `b2`: **"For the next three objects, you can use Assistant N. Ask it anything you like, as often as
  you like, then go and get the object."**

Each trial (from the participant's standing start at the room door):

1. **"Please find the {item}. Go."** — start the stopwatch on "Go".
2. Stop when the participant **touches** the item. If not found at the cap (180 s): **"Thanks, let's move on."**,
   record the cap and `success = 0`, and collect the item yourself. Each item is retrieved once, so found items
   simply go back on the tray (out of the way of the remaining hiding places).
3. Do not give hints. If asked "is it this one?", answer **"Keep going until you've got it in your hand."**
4. Record in `trials.csv` (or on the run sheet, transcribed the same day):
   `time_to_find_s`, `success`, `n_queries` (count of questions asked to the assistant; 0 for `none`),
   `assistant_correct` (1 if the assistant's first answer showed/named the right place; if Recall said it was
   unsure/abstained, 0), `assistant_confidence` (the % Recall displayed for its first answer ÷ 100; empty for
   `b2`/`none`), `followed_assistant_first` (1 if the first place they searched was the suggested one; empty if
   the assistant abstained or for `none`).

After the block's three trials, hand the block questionnaire (NASA-TLX; plus SUS + trust + usefulness for the
assistant blocks). Record in `questionnaires.csv` with the block's `condition` and `block_index`.

## 5. Debrief (5 min)

**"That's everything. The assistant you used as Assistant {N} was the system we're developing, and the other
one was a simpler comparison system. Do you have any questions?"**

- Optional preference tally (`questionnaires.md` §D).
- **Delete the participant's video and memory index now, in front of them** (Settings → Study → "Wipe all"),
  and tick "video deleted at hh:mm" on the run sheet. Remind them of the withdrawal route in the privacy notice.
- Fill `session_completed` (1/0).

## 6. After the session (same day)

- Transcribe paper forms into `study/data/*.csv` on the study laptop (FileVault on). Run
  `make study-analysis` — it validates the CSVs against `study/data_schemas.md` and reports errors per row.
- File the paper forms in the locked folder; shred the run sheet after transcription is checked.

## Pilot

Run 1–2 pilot sessions (IDs `X01`, `X02`, not analysed — the analysis only accepts `P\d\d`) to check timing,
the time cap (if > 20 % of `none` trials time out, raise it) and the hiding places. Then
`python -m recall.study.power` with variance components updated from the pilot.
