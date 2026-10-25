#!/usr/bin/env bash
# Create (once) and boot a headless arm64 API-35 emulator for instrumented tests / the e2e demo.
#   ./scripts/android_emulator.sh            # boot (no-op if a device is already attached)
#   ./scripts/android_emulator.sh --install  # also install emulator + system image via sdkmanager
# Prints the serial on the last line. Functional testing only: emulator numbers are never energy/latency results.
set -euo pipefail
SDK=${ANDROID_HOME:-${ANDROID_SDK_ROOT:-$HOME/Library/Android/sdk}}
AVD=${RECALL_AVD:-recall_api35}
IMG="system-images;android-35;google_apis;arm64-v8a"
export JAVA_HOME=${JAVA_HOME:-/opt/homebrew/opt/openjdk@17}
ADB="$SDK/platform-tools/adb"
SDKMANAGER=$(command -v sdkmanager || echo "$SDK/cmdline-tools/latest/bin/sdkmanager")
AVDMANAGER=$(command -v avdmanager || echo "$SDK/cmdline-tools/latest/bin/avdmanager")

if [[ "${1:-}" == "--install" || ! -x "$SDK/emulator/emulator" || ! -d "$SDK/system-images/android-35/google_apis/arm64-v8a" ]]; then
  yes | "$SDKMANAGER" --sdk_root="$SDK" "cmdline-tools;latest" "emulator" "platform-tools" "$IMG" >/dev/null
fi
if ! "$SDK/emulator/emulator" -list-avds | grep -qx "$AVD"; then
  echo no | "$AVDMANAGER" create avd -n "$AVD" -k "$IMG" -d pixel_6 --force >/dev/null
fi
# Already running? (match by AVD name so we never touch other people's emulators)
for s in $("$ADB" devices | awk '/emulator-/{print $1}'); do
  if [[ "$("$ADB" -s "$s" emu avd name 2>/dev/null | head -1 | tr -d '\r')" == "$AVD" ]]; then echo "$s"; exit 0; fi
done
LOG=${TMPDIR:-/tmp}/recall_emulator.log
nohup "$SDK/emulator/emulator" -avd "$AVD" -no-window -no-audio -no-boot-anim -gpu swiftshader_indirect -no-snapshot \
  -memory 4096 >"$LOG" 2>&1 &
echo "booting $AVD (log: $LOG) …" >&2
for _ in $(seq 1 180); do
  for s in $("$ADB" devices | awk '/emulator-/{print $1}'); do
    if [[ "$("$ADB" -s "$s" emu avd name 2>/dev/null | head -1 | tr -d '\r')" == "$AVD" ]] && \
       [[ "$("$ADB" -s "$s" shell getprop sys.boot_completed 2>/dev/null | tr -d '\r')" == "1" ]]; then
      "$ADB" -s "$s" shell settings put global window_animation_scale 0 || true
      "$ADB" -s "$s" shell settings put global transition_animation_scale 0 || true
      "$ADB" -s "$s" shell settings put global animator_duration_scale 0 || true
      echo "$s"; exit 0
    fi
  done
  sleep 2
done
echo "emulator did not boot in 6 min; see $LOG" >&2
exit 1
