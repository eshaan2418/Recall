PY ?= .venv/bin/python
export PYTHONPATH := $(CURDIR)

.PHONY: help setup test lint typecheck ci data-epic features eval-offline trigger memory calib compress \
        android android-test demo energy energy-device hoi-student export annotate-personal eval-personal standin-personal \
        baseline-vlm privacy-eval quick study-design study-sim study-power study-analysis reproduce-quick reproduce-full paper check-numbers scan

help:
	@grep -E '^[a-z-]+:' Makefile | cut -d: -f1 | sort | tr '\n' ' '; echo

setup:
	uv venv --python 3.12 .venv && uv pip install --python .venv -r requirements.txt

lint:
	$(PY) -m ruff check recall tests
typecheck:
	$(PY) -m mypy
test:
	$(PY) -m pytest tests -m "not slow"
ci: lint typecheck test

# ---------- data ----------
data-epic:
	./scripts/download_epic.sh
features:
	$(PY) -m recall.perception.extract
queries:
	$(PY) -m recall.bench.make_queries

# ---------- experiments (each writes results/<milestone>/...) ----------
eval-offline:
	$(PY) -m recall.bench.run_m1
trigger:
	$(PY) -m recall.bench.run_m2
memory:
	$(PY) -m recall.bench.run_m3
calib:
	$(PY) -m recall.bench.run_m4
PYX ?= .venv-export/bin/python
hoi-student:
	$(PY) -m recall.compress.hoi_student --seed 0
export:
	$(PYX) -m recall.compress.export
compress: export
	$(PYX) -m recall.bench.run_m5 --parts abcde
baseline-vlm:
	$(PY) -m recall.baselines.vlm
privacy-eval:
	$(PY) -m recall.privacy.eval && $(PY) -m recall.privacy.report
energy:
	$(PY) -m recall.energy.model
energy-device:
	$(PY) -m recall.energy.harness --minutes 60 --reps 3
annotate-personal:
	$(PY) -m recall.data.annotate.server
eval-personal:
	$(PY) -m recall.bench.run_m9
standin-personal:
	$(PY) -m recall.bench.run_m9 --standin 3
quick:
	$(PY) -m recall.bench.quick

# ---------- user study (M10) ----------
study-design:
	$(PY) -m recall.study.design
study-sim:
	$(PY) -m recall.study.simulate && $(PY) -m recall.study.analysis --simulated
study-power:
	$(PY) -m recall.study.power --reps 300 --n 10 12 15
study-analysis:
	$(PY) -m recall.study.analysis

# ---------- android ----------
android:
	cd android && ./gradlew assembleDebug assembleRelease
android-test:
	cd android && ./gradlew testDebugUnitTest
android-connected:
	cd android && ./gradlew connectedDebugAndroidTest
demo:
	./scripts/demo_e2e.sh

# ---------- reproduction / paper ----------
reproduce-quick:
	./scripts/reproduce_quick.sh
reproduce-full:
	./scripts/reproduce_full.sh
paper:
	$(PY) -m recall.bench.figures && cd paper && tectonic main.tex
check-numbers:
	$(PY) -m recall.bench.check_numbers
scan:
	./scripts/scan_repo.sh
