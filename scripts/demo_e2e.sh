#!/usr/bin/env bash
# Scripted end-to-end demo (make demo):
#   1. boots/uses an Android device (ANDROID_SERIAL, else the recall_api35 emulator via scripts/android_emulator.sh),
#   2. builds + installs the debug app and its test APK (models from models/litert are packaged if present),
#   3. injects the synthetic clip (android/app/src/androidTest/assets/e2e/keys_on_table.webm: a hand puts the keys on
#      the table) as camera input, lets the pipeline run, asks "Where are my keys?" and asserts "table":
#        - plumbing mode (synthetic heads/embedder; always runs, no model files needed)
#        - real-models mode (installed LiteRT embedder + exported heads/calibrator; needs models/litert)
#   4. with real models: also drives the capture service like the energy harness
#      (am start-foreground-service … -a ai.recall.START, files/counters.json, -a ai.recall.STOP).
# REQUIRE_MODELS=1 makes missing models an error instead of a skip.
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT=$PWD
SDK=${ANDROID_HOME:-${ANDROID_SDK_ROOT:-$HOME/Library/Android/sdk}}
export ANDROID_HOME=$SDK JAVA_HOME=${JAVA_HOME:-/opt/homebrew/opt/openjdk@17}
ADB="$SDK/platform-tools/adb"
REQUIRED=(gate.tflite embedder.tflite hoi_student.tflite heads.bin vocab.json calibrator.json answer_config.json)

missing=()
for f in "${REQUIRED[@]}"; do [[ -s "$ROOT/models/litert/$f" ]] || missing+=("$f"); done
[[ -s "$ROOT/models/litert/face_detector.tflite" ]] || echo "note: models/litert/face_detector.tflite missing → thumbnails are not stored (privacy fail-closed)."
if (( ${#missing[@]} )); then
  echo "=================================================================================================="
  echo " Real models not found in models/litert: ${missing[*]}"
  echo " Produce them with the Python export (see android/README.md, 'Asset contract')."
  echo " Running the PLUMBING demo only (synthetic heads/embedder); the real-model check is reported SKIPPED."
  echo "=================================================================================================="
  [[ "${REQUIRE_MODELS:-0}" == "1" ]] && { echo "REQUIRE_MODELS=1 → failing." >&2; exit 2; }
fi

if [[ -z "${ANDROID_SERIAL:-}" ]]; then
  ANDROID_SERIAL=$("$ROOT/scripts/android_emulator.sh" | tail -1)
fi
export ANDROID_SERIAL
echo "device: $ANDROID_SERIAL ($("$ADB" shell getprop ro.product.model | tr -d '\r'), API $("$ADB" shell getprop ro.build.version.sdk | tr -d '\r'))"

(cd android && ./gradlew --console=plain -q installDebug installDebugAndroidTest)

echo "--- instrumented e2e (ai.recall.E2eDemoTest) ---"
OUT=$("$ADB" shell am instrument -w -r -e class ai.recall.E2eDemoTest ai.recall.test/androidx.test.runner.AndroidJUnitRunner | tr -d '\r')
echo "$OUT" | grep -oE "E2E\[.*" || true
if ! echo "$OUT" | grep -q "INSTRUMENTATION_CODE: -1" || echo "$OUT" | grep -qE "FAILURES!!!|INSTRUMENTATION_FAILED|shortMsg=Process crashed"; then
  echo "$OUT" | grep -E "INSTRUMENTATION_STATUS: stack=|FAILURES|Process crashed" | head -20
  echo "E2E DEMO FAILED" >&2; exit 1
fi

if (( ${#missing[@]} == 0 )); then
  echo "--- capture service via adb (energy-harness interface) ---"
  CLIP=android/app/src/androidTest/assets/e2e/keys_on_table.webm
  "$ADB" push "$CLIP" /data/local/tmp/recall_e2e.webm >/dev/null
  "$ADB" shell "cat /data/local/tmp/recall_e2e.webm | run-as ai.recall sh -c 'mkdir -p files/e2e && cat > files/e2e/keys_on_table.webm'"
  "$ADB" shell pm grant ai.recall android.permission.CAMERA
  "$ADB" shell pm grant ai.recall android.permission.POST_NOTIFICATIONS || true
  "$ADB" shell run-as ai.recall rm -f files/counters.json
  "$ADB" shell am start-foreground-service -n ai.recall/.service.CaptureService -a ai.recall.START \
     --es policy all --ez backoff false --es replay_file /data/user/0/ai.recall/files/e2e/keys_on_table.webm >/dev/null
  for _ in $(seq 1 90); do
    C=$("$ADB" shell run-as ai.recall cat files/counters.json 2>/dev/null | tr -d '\r' || true)
    [[ "$C" == *'"recording":false'* ]] && break
    sleep 2
  done
  "$ADB" shell am start-foreground-service -n ai.recall/.service.CaptureService -a ai.recall.STOP >/dev/null
  echo "counters.json: $C"
  [[ "$C" == *'"events":'[1-9]* ]] || { echo "service replay produced no events" >&2; exit 1; }
fi
echo "E2E DEMO PASSED"
