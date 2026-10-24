from __future__ import annotations

import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.get("RECALL_DATA", REPO / "data"))
EPIC_RAW = DATA / "raw" / "epic"
EPIC_ANN = EPIC_RAW / "annotations"
CACHE = DATA / "cache"
RESULTS = REPO / "results"
MODELS = Path(os.environ.get("RECALL_MODELS", REPO / "models"))
FIXTURES = REPO / "tests" / "fixtures"
