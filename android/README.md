# Recall — Android app (M6) and on-device privacy layer (M7)

On-device visual memory: a foreground service watches through the back camera at 2 fps. A tiny always-on gate decides
when to run heavy perception: a hand–object student plus the MobileCLIP2-S2 image tower. Hand–object interactions
become *events* (take / put / touch) with embeddings, which are stored in an encrypted database. "Where are my keys?"
is answered from those events with a calibrated confidence and an evidence thumbnail, or the app says "I don't know".
No video is stored. The app has no network permission.

## Status: what is verified, and where

| Check | Where it ran | Result |
|---|---|---|
| `./gradlew testDebugUnitTest assembleDebug assembleRelease lint` (the CI command) | macOS host, JDK 17 | passes (lint: 0 errors) |
| 30 JVM unit tests, including Python-parity tests | host JVM | pass |
| 26 instrumented tests (Compose UI, service incl. live CameraX, SQLCipher, redaction, model contract, e2e) | **arm64 API-35 emulator** (headless, Apple-Silicon host) | pass |
| `scripts/demo_e2e.sh` (`make demo`), plumbing mode | emulator | passes: "Probably the table — … 95% confident." |
| `scripts/demo_e2e.sh`, real-model mode | — | **not run yet**: needs `models/litert` (see below) |
| Anything on a physical phone | — | **never run**. There are no on-device latency, energy, battery or thermal numbers here, and emulator timings must not be read as such. |

## Architecture

```
CameraX (2 fps, 456×256)  ─┐                      ┌─ debug only: ReplaySource (video file as camera input)
                           ▼                      ▼
CaptureService (foreground, camera type; ongoing notification + one-tap Pause; START_STICKY)
   │  Scheduler.plan(thermal status, battery %, charging) → stride / burst / threshold offset / pause  (backoff)
   ▼
Pipeline (single thread)
   StepPolicy: all | uniform-k | learned (LiteRtGate → BurstController)        core/Scheduler.kt, core/Policies.kt
   → LiteRtEmbedder(frame) → place clustering → "never record here" check
   → LiteRtHandObject → object / release-context crops → LiteRtEmbedder
   → EventExtractor (streaming port of recall/perception/events.py)            core/Events.kt
   → "never track" check → Redactor (BlazeFace + screen heuristic, fail-closed) privacy/
   → MemoryDb (SQLCipher; passphrase wrapped by an Android Keystore AES-GCM key) data/MemoryDb.kt
UI (Compose): Ask (text + on-device speech) · answer card (thumbnail, confidence, "Show me the moment",
   "Forget this", "Never track") · Timeline · Privacy (retention, places, objects, wipe)        ui/
QueryEngine: QueryParser → Answerer (latest/best/recency) → Calibrator (logistic + isotonic) → Renderer  core/
```

* `core/` is pure Kotlin, a port of the Python reference (`recall/qa/*`, `recall/perception/{events,heads}.py`,
  `recall/trigger/policies.py`). It is checked by `ParityTest` against JSON fixtures exported from Python
  (`scripts/android_fixtures.sh` regenerates them in `app/src/test/resources/parity/`).
* **Robustness.** All memory and settings (retention, blocked places and objects, "recording wanted", operating point)
  live in the DB. The ViewModel survives rotation; the query and tab are kept in `SavedStateHandle` across process
  death. After process death, a sticky restart resumes only if the user had recording on. If camera permission is
  missing or revoked, the service stops with a status message instead of crashing. Camera errors stop capture
  cleanly. Camera frames are dropped while the pipeline is busy (keep-latest).
* **Thermal / battery backoff** (`Scheduler`, unit-tested):

  | Condition | Effect |
  |---|---|
  | thermal LIGHT | gate threshold +0.05 |
  | thermal MODERATE | trigger every 2nd step, burst 3, threshold +0.1 |
  | thermal SEVERE | every 3rd step, burst 2, threshold +0.2 |
  | thermal CRITICAL or above, or battery ≤ 8 % while not charging | capture paused |
  | battery ≤ 20 % while not charging | stride ≥ 2, burst ≤ 3, threshold +0.1 |

* **Privacy (M7, on-device half).**
  * The merged release manifest has no `INTERNET` or other network permission. This is enforced by the
    `verifyReleaseManifestPolicy` task (runs before `assembleRelease` and in `check`) and by `ManifestPolicyTest`.
  * Backup and device transfer are disabled.
  * The DB is encrypted with SQLCipher. Its 32-byte random passphrase is stored only AES-256-GCM-wrapped by a
    non-exportable Keystore key, so deleting the key erases the memory cryptographically.
  * Faces (BlazeFace, low threshold + 25 % padding) and bright text or screen regions (heuristic) are pixelated
    **before** a thumbnail is stored. Without the face model, or if a detector throws, no thumbnail is stored at all
    (fail-closed). Raw frames are never written.
  * Retention: auto-delete after 1 / 7 (default) / 30 days, or never. It runs at app start and hourly while capturing.
  * "Never record here" blocks the current place cluster: nothing is stored there, and past events at that place are
    deleted. "Never track this object" drops future events whose top class is that object and deletes existing ones.
  * Speech uses `SpeechRecognizer.createOnDeviceSpeechRecognizer` with `EXTRA_PREFER_OFFLINE`. If on-device
    recognition is unavailable, the app asks you to type. It deliberately does not fall back to a possibly-cloud
    recognizer.

## Build / install / test

Requirements: JDK 17, an Android SDK with `platforms;android-37.0`, because the current AndroidX/CameraX releases
require compileSdk ≥ 37. targetSdk is 35 and minSdk is 29.

```bash
cd android
./gradlew testDebugUnitTest assembleDebug assembleRelease lint      # what CI runs
./gradlew installDebug                                              # needs a device / emulator
./gradlew connectedDebugAndroidTest                                 # instrumented tests (set ANDROID_SERIAL if >1 device)
../scripts/android_emulator.sh                                      # create + boot headless recall_api35 AVD, prints serial
../scripts/demo_e2e.sh                                              # == make demo
```

Model assets: if `../models/litert/` exists, its non-empty `*.tflite`, `*.json` and `*.bin` files are copied into the
APK assets (`copyModelAssets` task). Without them the app still builds, the unit tests pass, and the UI shows a
yellow **"Fallback mode — Models not installed …"** banner with Start/Ask disabled. The Timeline and Privacy screens
still work.

Debug-only affordances. Release builds have none of these: `src/release/.../DebugHooks.kt` returns no replay source,
and the release service is not exported.

* `ReplaySource`: a video file as camera input, via the `replay_file` extra.
* `CaptureService` is exported and guarded by `android.permission.DUMP`. The adb shell holds that permission and
  ordinary apps can't get it. The energy harness drives the service this way:
  ```bash
  adb shell am start-foreground-service -n ai.recall/.service.CaptureService -a ai.recall.START \
      --es policy {all|uniform|learned} --ei k 5 --ez backoff {true|false}
  adb shell run-as ai.recall cat files/counters.json   # rewritten every 10 s
  adb shell am start-foreground-service -n ai.recall/.service.CaptureService -a ai.recall.STOP
  ```
  `counters.json` contains `steps, gated, heavy, events, db_bytes, thermal_status, backoff_level`, plus `policy, k,
  backoff, recording, t` and privacy counters. `run-as` requires a debuggable build, so energy runs use the debug
  APK.

### E2E demo

`android/tools/make_test_video.py` (`scripts/android_make_test_video.sh`) generates the committed synthetic clip
`app/src/androidTest/assets/e2e/keys_on_table.webm`: 16 s, VP8, drawn shapes only, no people and no EPIC frames. A
skin-coloured hand carries a yellow "KEYS" object from a grey COUNTER and puts it on a brown TABLE. The `.webm`
container is used because `*.mp4` is git-ignored outside `tests/fixtures`.

`E2eDemoTest` decodes the clip at 2 fps through `ReplaySource` → `Pipeline` → SQLCipher → `QueryEngine`, asks
"Where are my keys?" and asserts the answer *table*. It also asks about a never-seen object and asserts an
abstention. It runs in two modes:

* **plumbing** (always runs): synthetic stand-ins (always-on gate, colour-histogram embedder, matching heads). This
  verifies the device path end to end, not model accuracy.
* **real models**: the installed `embedder.tflite` and exported heads/calibrator. The HOI student is replaced by a
  colour-key hand detector, because the clip's hand is drawn and a real HOI model can't be expected to see it.
  `-e gate model` uses `gate.tflite`. The test is reported as SKIPPED when the models are missing.

With real models present, `demo_e2e.sh` also replays the clip through the adb-driven service and checks
`counters.json`.

Emulator note: on the arm64 emulator under an Apple-Silicon hypervisor, XNNPACK's CPU probe dies with SIGILL. So
`LiteRtModel` disables XNNPACK when it detects an emulator. Real phones keep XNNPACK, and the GPU delegate falls back
to CPU when it is unavailable.

## Asset contract (`models/litert/` → APK assets)

All `.tflite` models must have **float32 input/output tensors**: fp32 or int8-dynamic / int4 weights. Static-int8
exports with INT8 I/O are rejected at load with a clear error. Tensors are identified by shape where possible, and
4-D image inputs may be NCHW or NHWC. `ModelContractTest` (instrumented) loads every installed model through the
production wrappers and runs one inference.

| File | Inputs | Outputs | Pre-processing |
|---|---|---|---|
| `gate.tflite` | `x` [1,8,9,36,64] f32; `imu` [1,8,7] f32 (optional 2nd input, found by rank) | 1 element ([1] or [1,1]): **logit** (the app applies sigmoid) | 36×64 bilinear thumbnail of the 456×256 frame, RGB/255; channels = `f-0.5`, `4·(f-f[t-1])`, `4·(f-f[t-2])`; context = last 8 gate steps, oldest→newest, left-padded by repeating the first step (`recall/trigger/gate.py`). IMU = zeros (flag 0); phone IMU is not wired yet |
| `embedder.tflite` | [1,3,S,S] NCHW (or [1,S,S,3]) f32, S read from the model (256) | [1,D] f32 (re-L2-normalised on device) | shortest side → S (bilinear), centre crop, RGB `(px/255 − mean)/std` with mean 0 / std 1 by default (dfndr2b). Override with `answer_config.json → models.embedder.{mean,std}` |
| `hoi_student.tflite` | [1,3,H,W] NCHW (or NHWC) f32, H×W read from the model (144×256); whole 456×256 frame resized (not cropped), RGB 0..1 | [1,2,14] f32 per side (0 = left, 1 = right): `[0]` hand-present logit, `[1..5]` state logits (none, self, other, portable, stationary), `[6..9]` hand box l,t,r,b, `[10..13]` object box l,t,r,b, both in 0..1 (already sigmoided) | present if σ(logit) ≥ 0.5 (`models.hoi.present_thresh`). Contact = state 3 (portable) or 4 (stationary). Object crops are expanded ×1.15 and release-context crops ×3, like `extract.py::expand_box` |
| `face_detector.tflite` | [1,128,128,3] NHWC f32 in [−1,1] | [1,896,16] regressors and [1,896,1] score logits (matched by shape) | MediaPipe BlazeFace **short-range** (`face_detection_short_range.tflite`, Apache-2.0). Threshold 0.4 (`models.face.thresh`) |

Query bundle (written by the Python exporter):

* **`heads.bin`**: little-endian. The object head, then the location head. Each head is:
  `int32 C, int32 D, float32 scale, float32 alpha, float32[C·D] text bank (row-major, L2-normalised),
  int32 K, int32[K] probe classes, float32[K·D] probe W, float32[K] probe b` (K = 0 means no probe).
  * Semantics follow `BlendHead.log_probs`: `alpha·log(softmax(scale·⟨e,bank⟩)+1e-9) + (1-alpha)·lp`, where
    `lp = log(softmax(W·e+b)+1e-9)` on the probe classes and `log 1e-4` elsewhere.
  * A **binary** sklearn probe (`coef_` shape [1,D]) must be exported as two rows, `W=[0, coef]` and
    `b=[0, intercept]`, so that softmax equals sigmoid.
  * D must equal the embedder output dim. The object head has C = len(nouns); the location head has C = len(loc_names).
* **`vocab.json`**:
  `{"nouns": [C names], "synonyms": {"text": noun_class}, "loc_ids": [noun ids of the location classes], "loc_names": [names]}`.
  * `synonyms` should hold the `recall/qa/resolve.py::_synonyms()` table.
  * Lookup tries `t`, `t.rstrip("s")`, and `t[:-2]` when `t` ends in "es". There is no text encoder on device.
* **`calibrator.json`**:
  `{"features": [...], "coef": [...], "intercept": x, "iso_x": [...], "iso_y": [...], "threshold": t}`.
  * `features` are named, in `feat_matrix` order: `score, is_argmax, score_margin, loc_p, loc_margin, log_n_obs,
    log_age_s, n_conflicts, second_score, release_obs, hand_score, n_candidates, fallback, p_on, p_on_margin,
    kind=where, kind=temporal, kind=state`.
  * `iso_x` / `iso_y` are `IsotonicRegression.X_thresholds_` / `y_thresholds_`; leave both empty for the logistic
    method.
  * `threshold` is the abstain-below confidence (`threshold_from_val`).
* **`answer_config.json`**:
  `{"configs": {"where"|"temporal"|"state": {"mode": "latest|best|recency", "tau", "recency_s", "score": "prob|ratio",
  "fallback_take": true}}, "gate_threshold": thr, "state_prior": {"cls": p_on}, "global_state_prior": 0.5,
  "state_probe": {"coef": [2D floats over (loc_emb, scene_emb)], "intercept": b}, "models": {"embedder": {"mean",
  "std"}, "hoi": {"present_thresh"}, "face": {"thresh"}}}`. Everything except `configs` is optional.

Live burst length is 4 (`run_m2.BURST`), reduced by backoff. Scene / place clustering uses the frame embedding with
cosine ≥ 0.80.

## Layout

```
app/src/main/java/ai/recall/{core,ml,privacy,data,service,ui}   app code (core/ is pure Kotlin)
app/src/debug/…   ReplaySource + exported/DUMP-guarded service        app/src/release/…  no-op hooks
app/src/test/…    JVM tests + parity fixtures                         app/src/androidTest/…  instrumented tests + e2e clip
tools/make_parity_fixtures.py, tools/make_test_video.py              (driven by ../scripts/android_*.sh)
```
