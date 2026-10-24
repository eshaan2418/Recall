PY ?= .venv/bin/python
export PYTHONPATH := $(CURDIR)

.PHONY: help setup test lint typecheck ci data-epic features eval-offline trigger memory calib compress \
        android android-test demo energy reproduce-quick reproduce-full paper check-numbers scan clean-results

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
compress:
	$(PY) -m recall.bench.run_m5
energy:
	$(PY) -m recall.energy.model

# ---------- android ----------
android:
	cd android && ./gradlew assembleDebug assembleRelease
android-test:
	cd android && ./gradlew testDebugUnitTest
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
