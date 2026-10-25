# Recall threat model (M7)

Recall is a research prototype: an on-device visual memory over egocentric camera frames. This document says
what the prototype protects, what it does not, and what a production version would still need. It describes the
code in this repo as of M7. Statements about the app point to the Kotlin source under `android/`. Redaction numbers
live in [`README.md`](README.md) and come from `results/m7/redaction.json`.

## 1. What is stored, and where

| Data | Stored? | Where / form |
|---|---|---|
| Camera video / full frames | **Never.** Frames are processed in memory and dropped. | – |
| Audio | **Never stored.** The voice query uses `SpeechRecognizer.createOnDeviceSpeechRecognizer` (API 31+, offline preferred). | – |
| Event records (time, kind, hand side, object/location top-k, box, place id) | Yes | `events` table, SQLCipher DB |
| Embeddings (object crop, location crop, whole scene; fp16) | Yes | same table (`obj_emb`, `loc_emb`, `scene_emb`) |
| One thumbnail per event (crop around the hand-object box ×3, 160×90 JPEG q70, ≤4 KB) | Yes, **only after redaction** | `thumb` column |
| Place centroids, place names, blocked flags, never-track classes, settings | Yes | `places`, `blocked_objects`, `settings` tables |
| Pipeline counters (counts only, no content) | Yes | `files/counters.json` (energy harness) |

## 2. Controls in the prototype

**No network.** The merged *release* manifest has no `INTERNET` or `ACCESS_NETWORK_STATE` permission. Both are
explicitly removed with `tools:node="remove"`, so a library cannot merge them back in. `ManifestPolicyTest`
(`android/app/src/test/java/ai/recall/ManifestPolicyTest.kt`) parses the merged release manifest and fails the build
if a forbidden permission appears. It also checks that `CaptureService` is not exported and that backup is off.
Without `INTERNET`, an app process cannot open sockets, so the app cannot exfiltrate memory by itself.

**No backup / transfer.** `android:allowBackup="false"`, `fullBackupContent="false"`, and `data_extraction_rules.xml`
excludes every domain from cloud backup and device-to-device transfer.

**Encryption at rest.** The memory DB is SQLCipher. Its 32-byte random passphrase is stored only in wrapped form,
encrypted with AES-256-GCM under a non-exportable Android Keystore key (`KeyManager` in `MemoryDb.kt`). The wrapped
blob lives in `noBackupFilesDir`. Deleting the Keystore entry (`KeyManager.destroy`) cryptographically erases the
memory.

**Redaction before storage, fail closed.** `Pipeline` calls `Redactor.thumbnail` and stores only what it returns.
`Redactor` (`android/.../privacy/Redactor.kt`) pixelates every box from BlazeFace short-range (faces) and from a
screen/document heuristic. The mosaic cell is `max(6, min(box)/6)` px. The image is then downscaled to 160×90. The
redactor returns **no thumbnail** in three cases: the face model is missing, the face detector is absent from the
detector list, or any detector throws. The event is then stored without an image (`thumbsWithheld` counter). The
Python port `recall/privacy/redact.py` mirrors this logic, and `tests/test_privacy.py` checks the fail-closed
behaviour.

**User controls.**
- *Retention*: events older than N days are deleted (default 7). This runs when the service starts, hourly while it
  runs, at app start, and when the setting changes (`CaptureService`, `RecallRuntime.applyRetention`).
- *Never-record zones*: marking a place as blocked deletes its existing events. From then on, while place
  recognition says you are in that place, nothing is extracted, and any open bout is discarded (`Pipeline`).
- *Never-track objects*: blocking an object class deletes its events and drops future events whose top class is that
  object.
- *Delete all*: `wipe()` deletes events and places, then runs `VACUUM`.

**Research-side data hygiene.**
- Cloud baselines are off by default. The VLM baseline (B1, `recall/baselines/vlm.py`) runs an open model locally. A
  frontier-API mode is reserved behind an explicit `--allow-cloud` flag, which is meant only for *public*
  EPIC-KITCHENS frames. That mode is not implemented and is not used for any reported number. Personal recordings
  (`data/personal/`) must never be sent to a cloud model.
- `scripts/scan_repo.sh` runs gitleaks over all refs and the working tree. It also scans every blob in history plus
  untracked files, by extension *and* magic bytes, and fails on images, video, audio, arrays, weights, archives,
  databases, blobs over 2 MB, and personal-data paths. The only exceptions are an explicit allow-list (synthetic test
  fixtures, plots, launcher icons). It also runs BlazeFace over every allow-listed image and fails if it finds a
  face. Evaluation datasets (WIDER FACE, EPIC) and models live in git-ignored `data/raw/` and `models/`.

## 3. Adversaries

| Adversary | Outcome in the prototype |
|---|---|
| **Network attacker / cloud provider** | Nothing to intercept: the app has no network permission and no backup. |
| **Thief with a locked phone** | Data is protected by Android file-based encryption plus SQLCipher, and the Keystore key is not exportable. This is strongest before first unlock after boot. After first unlock, protection rests on the lock screen: if the attacker can run code as the app, the Keystore key is usable (see below). |
| **Attacker holding the unlocked phone** (partner, coworker, border search, thief after unlock) | **Not protected.** They can open Recall and browse the timeline and thumbnails, or ask "where are my keys". The Keystore key is created without `setUserAuthenticationRequired` / `setUnlockedDeviceRequired`, so the DB opens whenever the app runs. The prototype has no app lock. |
| **Attacker who runs code as Recall's UID or as root** (rooted or compromised device, malicious debug build, `adb` on a debuggable build; debug builds also export `CaptureService`, guarded by `android.permission.DUMP` so only the adb shell can start it) | **Not protected.** They can call the Keystore as the app, unwrap the passphrase, and read the DB. They can also read live frames from the camera service. |
| **Malicious app on the same phone** (no root) | The sandbox blocks access to the app's private files and Keystore entry. No exported services or providers: `CaptureService` is `exported=false`, and only the launcher activity is exported. Residual risks: screen capture or accessibility abuse while the Recall UI is open (no `FLAG_SECURE`), and side channels such as the camera-in-use indicator. |
| **Malicious or compromised library inside Recall** | It runs with the app's permissions, which do *not* include network. It could still write data somewhere world-readable or misuse the camera. No dependency audit yet. |
| **Someone with the researcher's laptop or the repo** | The repo holds no frames, faces, or personal data (enforced by the scan). Downloaded datasets and models are git-ignored but sit unencrypted on disk. `data/personal/` relies on the laptop's disk encryption. |

## 4. Bystanders and consent

A head-worn or chest-worn camera captures people who did not agree to be recorded. In the prototype:

- No video or audio is kept. A bystander can persist only as (a) a redacted thumbnail, (b) event metadata, or (c) an
  embedding of a frame they appeared in.
- Faces are pixelated **when the detector finds them**. Detection is imperfect, especially for small, profile,
  occluded, or blurred faces. See the size-bucketed recall in [`README.md`](README.md).
- The app gives bystanders no notice. A foreground-service notification is shown to the *wearer* only.
- The user study (`study/run_session.md`, `study/data_handling.md` §5) runs in a room with no other people and no
  screens, documents, or photos of people in view. If someone enters, recording stops and that segment is deleted.
  That is a procedural control, not a technical one, and it covers the study only, not general use.

## 5. Residual risks (known, not mitigated)

1. **Missed faces.** BlazeFace short-range is built for faces filling a large part of a selfie-style frame. The
   redactor squashes the whole crop to 128×128 with no tiling. Faces under about 10% of the crop height are mostly
   missed. Profile, occluded, and motion-blurred faces are missed more often. Pixelation also slows recognition
   rather than preventing it: coarse mosaics of large faces can still be matched by familiar viewers or by
   de-pixelation models.
2. **Text on objects and dark screens.** The screen/document heuristic flags only *bright* cells with dense edges.
   Dark-mode screens, small labels, handwriting, envelopes, and medication names often pass through. Bright tiles and
   printed packaging get over-redacted. Numbers are in `README.md` (synthetic set).
3. **Location inference from thumbnails.** Thumbnails of the home interior (window views, mail, distinctive decor)
   can reveal where the user lives. Place clusters and their names reveal the home layout and daily routine.
4. **Timeline as a behavioural record.** Even without images, timestamped events ("took pills 08:02", "left keys at
   the door") form a sensitive activity log. Retention limits how far back it reaches, but does not change what it
   contains.
5. **Embeddings are not redacted and are partly invertible.** Embeddings are computed from the *unredacted* frame and
   crops, *before* redaction. CLIP-family embeddings preserve scene semantics, readable text ("typographic"
   content), and some identity-correlated information. Published inversion and attribute-inference attacks recover
   coarse content from such vectors. Anyone who can read the DB gets this as well as the thumbnails.
6. **Never-record zones depend on place recognition.** A blocked place is detected from scene embeddings. Frames from
   the first moments in a room, or from a misrecognised room, can still produce events.
7. **Deletion semantics.** `wipe()` keeps settings and never-track classes. SQLite free pages are encrypted but
   not overwritten with `secure_delete`. Flash wear-levelling means only key destruction is a real erase.
8. **Keystore without user authentication.** See "unlocked phone" above. A per-session biometric gate is not
   implemented.
9. **Port and evaluation gaps.** The desktop evaluation runs a Python port. Resizing and JPEG encoding differ
   slightly from Android's Skia, which can flip borderline detections. The face numbers come from WIDER FACE, which is
   third-person web photos, **not egocentric** footage. The screen numbers come from a **synthetic** set.

## 6. What a production version would need

- **Stronger, measured redaction.** Use a full-range face detector on tiles, plus a person/body detector. Redact by
  solid fill or heavy blur rather than mosaic. Use a learned text/screen detector (dark-mode aware). Validate recall
  on consented egocentric data from the target device, with an explicit recall target per face-size bucket and a CI
  regression gate.
- **Redact before embedding,** or embed only crops that contain no detected people or screens. Consider storing
  quantised or projected embeddings to limit inversion. Evaluate inversion attacks on the stored representation.
- **Access control.** App lock with biometric or device credential. Keystore key with
  `setUserAuthenticationRequired` (time-bounded) and `setUnlockedDeviceRequired`, StrongBox where available.
  `FLAG_SECURE` on screens that show thumbnails.
- **Erasure.** Make the "delete everything" action destroy the Keystore key and recreate the DB, and drop settings and
  places. Document what retention guarantees.
- **Bystander protections.** Visible capture indicator on the wearable. Automatic pause when faces are detected for
  longer than a set time. A no-thumbnail mode. A clear consent and notice policy.
- **Supply chain.** Pin and audit dependencies, verify model hashes at load (the Python side already pins
  `face_detector.tflite` by sha256), and use a reproducible release build.
- **Independent review.** Get an external security assessment and a DPIA before any deployment beyond the research
  study.
