"""M8 — on-device energy / thermal / storage harness (requires a phone on adb; produces `on-device` results).

Protocol per operating point (repeat ``--reps`` times, default 3 × 60 min):
  1. fixed brightness (screen off during capture; the foreground-service notification stays), airplane mode on
     (the app has no network permission anyway), charger disconnected (``dumpsys battery unplug``),
  2. ``dumpsys batterystats --reset``; start Perfetto (power rails + thermal + CPU freq if the device exposes them),
  3. start the app's capture service with the operating point via ``am start-foreground-service`` extras,
  4. sample ``dumpsys battery`` (level, voltage, temperature), ``dumpsys thermalservice`` (status + skin/SoC temps)
     and the app's own counters (frames gated / heavy invocations / events / DB bytes) every 30 s,
  5. stop; pull ``batterystats --checkin`` and the Perfetto trace; ``dumpsys battery reset``.

Operating points: always-heavy, uniform-k (matched budget), learned trigger, learned trigger + thermal backoff.

    python -m recall.energy.harness --minutes 60 --reps 3
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import time
from pathlib import Path

from recall.bench.common import result_header, write_result
from recall.paths import RESULTS

PKG = "ai.recall"
SERVICE = f"{PKG}/.service.CaptureService"
OPERATING_POINTS = {
    "always-heavy": {"policy": "all", "backoff": False},
    "uniform-k5": {"policy": "uniform", "k": 5, "backoff": False},
    "learned": {"policy": "learned", "backoff": False},
    "learned+backoff": {"policy": "learned", "backoff": True},
}
PERFETTO_CFG = """
buffers { size_kb: 65536 fill_policy: RING_BUFFER }
data_sources { config { name: "android.power" android_power_config { battery_poll_ms: 1000
  collect_power_rails: true battery_counters: BATTERY_COUNTER_CHARGE battery_counters: BATTERY_COUNTER_CURRENT
  battery_counters: BATTERY_COUNTER_VOLTAGE } } }
data_sources { config { name: "linux.sys_stats" sys_stats_config { thermal_period_ms: 1000 cpufreq_period_ms: 1000 } } }
duration_ms: DURATION
"""


def adb(*args: str, check: bool = True) -> str:
    return subprocess.run(["adb", *args], capture_output=True, text=True, check=check).stdout


def device_info() -> dict:
    g = lambda p: adb("shell", "getprop", p).strip()  # noqa: E731
    return {
        "model": g("ro.product.model"),
        "soc": g("ro.soc.model"),
        "android": g("ro.build.version.release"),
        "build": g("ro.build.fingerprint"),
    }


def battery() -> dict:
    s = adb("shell", "dumpsys", "battery")

    def f(k: str) -> int | None:
        m = re.search(rf"{k}: (-?\d+)", s)
        return int(m.group(1)) if m else None

    return {"level": f("level"), "voltage_mV": f("voltage"), "temp_dC": f("temperature")}


def thermal() -> dict:
    s = adb("shell", "dumpsys", "thermalservice", check=False)
    st = re.search(r"Thermal Status: (\d)", s)
    temps = dict(re.findall(r"mName=(\w+).*?mValue=([\d.]+)", s))
    return {"status": int(st.group(1)) if st else None, "temps_C": {k: float(v) for k, v in temps.items()}}


def app_counters() -> dict:
    s = adb("shell", "run-as", PKG, "cat", "files/counters.json", check=False)
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        return {}


def run_point(name: str, op: dict, minutes: float, out_dir: Path) -> dict:
    adb("shell", "cmd", "connectivity", "airplane-mode", "enable", check=False)
    adb("shell", "dumpsys", "battery", "unplug")
    adb("shell", "dumpsys", "batterystats", "--reset")
    cfg = PERFETTO_CFG.replace("DURATION", str(int(minutes * 60e3)))
    trace = f"/data/misc/perfetto-traces/recall_{name}.pftrace"
    subprocess.Popen(
        ["adb", "shell", "perfetto", "--txt", "-c", "-", "-o", trace, "--background"], stdin=subprocess.PIPE, text=True
    ).communicate(cfg)
    extras = []
    for k, v in op.items():
        extras += ["--es" if isinstance(v, str) else "--ez" if isinstance(v, bool) else "--ei", k, str(v).lower()]
    adb("shell", "am", "start-foreground-service", "-n", SERVICE, "-a", f"{PKG}.START", *extras)
    t0, samples = time.time(), []
    while time.time() - t0 < minutes * 60:
        samples.append({"t": time.time() - t0, **battery(), "thermal": thermal(), "app": app_counters()})
        time.sleep(30)
    adb("shell", "am", "start-foreground-service", "-n", SERVICE, "-a", f"{PKG}.STOP")
    stats = adb("shell", "dumpsys", "batterystats", "--checkin")
    (out_dir / f"{name}_batterystats.txt").write_text(stats)
    adb("pull", trace, str(out_dir / f"{name}.pftrace"), check=False)
    adb("shell", "dumpsys", "battery", "reset")
    hours = (samples[-1]["t"] - samples[0]["t"]) / 3600 if len(samples) > 1 else float("nan")
    drop = samples[0]["level"] - samples[-1]["level"] if samples else float("nan")
    app = samples[-1]["app"] if samples else {}
    return {
        "operating_point": name,
        "minutes": minutes,
        "battery_pct_per_hour": drop / hours if hours else None,
        "max_temp_dC": max(s["temp_dC"] or 0 for s in samples),
        "max_thermal_status": max(s["thermal"]["status"] or 0 for s in samples),
        "frames_processed": app.get("heavy"),
        "db_bytes_per_hour": (app.get("db_bytes", 0) / hours) if hours else None,
        "samples": samples,
        "note": "battery level has 1 % resolution; use the Perfetto charge counter for average power",
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes", type=float, default=60)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--points", nargs="*", default=list(OPERATING_POINTS))
    a = ap.parse_args()
    if "device" not in adb("devices", check=False).split("\n", 1)[-1]:
        raise SystemExit("no adb device: energy stays `estimated` (python -m recall.energy.model). See HUMAN_TODO.md")
    out_dir = RESULTS / "m8" / "device_raw"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = [run_point(f"{p}_r{r}", OPERATING_POINTS[p], a.minutes, out_dir) for r in range(a.reps) for p in a.points]
    write_result(
        "m8/energy_device.json", {**result_header("on-device", None, vars(a)), "device": device_info(), "rows": rows}
    )


if __name__ == "__main__":
    main()
