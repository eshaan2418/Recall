"""CSV schemas for the M10 user-study data (see ``study/data_schemas.md`` for the human-readable version).

Three files live in ``study/data/`` (git-ignored, on the study laptop only):

* ``participants.csv``   — one row per pseudonymous participant (no names, no contact details)
* ``trials.csv``         — one row per retrieval trial (9 per participant)
* ``questionnaires.csv`` — one row per participant x condition block (NASA-TLX; SUS/trust/usefulness for
  the two assistant conditions)

An optional ``simulated`` column (0/1) is allowed in every file; the real-data analysis refuses to run if any
row is marked simulated.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

PID_PATTERN = r"^P\d{2}$"
ASSISTANT_CONDITIONS = ("recall", "b2")
TLX_SCALES = ("mental", "physical", "temporal", "performance", "effort", "frustration")
SUS_ITEMS = 10
TRUST_ITEMS = 3
USEFUL_ITEMS = 2


@dataclass(frozen=True)
class Col:
    name: str
    kind: str  # "str" | "int" | "float" | "cat"
    required: bool = True  # False -> may be empty (NA)
    lo: float | None = None
    hi: float | None = None
    allowed: tuple[str, ...] | None = None
    pattern: str | None = None
    desc: str = ""
    na_unless: tuple[str, ...] | None = None  # must be NA unless condition in this tuple, and present if so


PARTICIPANTS: tuple[Col, ...] = (
    Col("participant_id", "str", pattern=PID_PATTERN, desc="Pseudonymous ID P01..P99 (key in the paper link sheet)"),
    Col("design_row", "int", lo=0, hi=98, desc="Row of the counterbalancing design (from assignments.csv)"),
    Col(
        "age_band",
        "cat",
        allowed=("18-24", "25-34", "35-44", "45-54", "55-64", "65+", "prefer_not"),
        desc="Coarse age band only",
    ),
    Col("tech_familiarity", "int", lo=1, hi=5, required=False, desc="Self-rated smartphone familiarity 1-5"),
    Col("familiar_with_space", "int", lo=0, hi=1, desc="1 if the participant knew the study room beforehand"),
    Col("session_completed", "int", lo=0, hi=1, desc="1 if all 3 blocks were completed"),
)

TRIALS: tuple[Col, ...] = (
    Col("participant_id", "str", pattern=PID_PATTERN),
    Col("trial_index", "int", lo=1, hi=99, desc="1-based position of the trial in the session"),
    Col("block_index", "int", lo=1, hi=3, desc="1-based block (condition) position"),
    Col("condition", "cat", allowed=("recall", "none", "b2")),
    Col("item", "str", desc="Item name exactly as in study_config.yaml"),
    Col("hiding_place", "str", desc="Hiding place name exactly as in study_config.yaml"),
    Col("time_to_find_s", "float", lo=0, desc="Seconds from item named to item touched; = time cap if not found"),
    Col("success", "int", lo=0, hi=1, desc="1 if found within the time cap"),
    Col("n_queries", "int", lo=0, required=False, desc="Queries typed/spoken to the assistant (0 for none)"),
    Col(
        "assistant_correct",
        "int",
        lo=0,
        hi=1,
        required=False,
        na_unless=ASSISTANT_CONDITIONS,
        desc="1 if the assistant's first answer showed the right place; NA for none",
    ),
    Col(
        "assistant_confidence",
        "float",
        lo=0,
        hi=1,
        required=False,
        na_unless=("recall",),
        desc="Confidence Recall displayed for its first answer (0-1); NA for none/b2",
    ),
    Col(
        "followed_assistant_first",
        "int",
        lo=0,
        hi=1,
        required=False,
        na_unless=ASSISTANT_CONDITIONS,
        desc="1 if the participant searched the suggested place first; NA for none",
    ),
)

_TLX = tuple(
    Col(f"tlx_{s}", "int", lo=0, hi=100, desc=f"NASA-TLX {s} 0-100 (steps of 5; performance: 0 = perfect)")
    for s in TLX_SCALES
)
_SUS = tuple(
    Col(f"sus_{i}", "int", lo=1, hi=5, required=False, na_unless=ASSISTANT_CONDITIONS, desc="SUS item 1-5")
    for i in range(1, SUS_ITEMS + 1)
)
_TRUST = tuple(
    Col(f"trust_{i}", "int", lo=1, hi=7, required=False, na_unless=ASSISTANT_CONDITIONS, desc="Trust Likert 1-7")
    for i in range(1, TRUST_ITEMS + 1)
)
_USEFUL = tuple(
    Col(f"useful_{i}", "int", lo=1, hi=7, required=False, na_unless=ASSISTANT_CONDITIONS, desc="Usefulness 1-7")
    for i in range(1, USEFUL_ITEMS + 1)
)
QUESTIONNAIRES: tuple[Col, ...] = (
    Col("participant_id", "str", pattern=PID_PATTERN),
    Col("condition", "cat", allowed=("recall", "none", "b2")),
    Col("block_index", "int", lo=1, hi=3),
    *_TLX,
    *_SUS,
    *_TRUST,
    *_USEFUL,
)

SCHEMAS: dict[str, tuple[Col, ...]] = {
    "participants": PARTICIPANTS,
    "trials": TRIALS,
    "questionnaires": QUESTIONNAIRES,
}
OPTIONAL_EXTRA = ("simulated",)


class SchemaError(ValueError):
    pass


def validate(df: pd.DataFrame, table: str) -> list[str]:
    """Return a list of human-readable problems (empty list = valid)."""
    cols = SCHEMAS[table]
    errs: list[str] = []
    names = [c.name for c in cols]
    missing = [n for n in names if n not in df.columns]
    if missing:
        errs.append(f"{table}: missing columns {missing}")
    extra = [c for c in df.columns if c not in names and c not in OPTIONAL_EXTRA]
    if extra:
        errs.append(f"{table}: unknown columns {extra}")
    for c in cols:
        if c.name not in df.columns:
            continue
        s = df[c.name]
        na = s.isna() | (s.astype(str).str.strip() == "")
        if c.required and na.any():
            errs.append(f"{table}.{c.name}: {int(na.sum())} empty value(s) in a required column")
        if c.na_unless is not None and "condition" in df.columns:
            applies = df["condition"].isin(c.na_unless)
            bad_present = (~applies & ~na).sum()
            if bad_present:
                errs.append(f"{table}.{c.name}: {int(bad_present)} value(s) given where it must be NA")
            if c.name.startswith(("sus_", "trust_", "useful_")):
                bad_missing = (applies & na).sum()
                if bad_missing:
                    errs.append(f"{table}.{c.name}: {int(bad_missing)} missing value(s) for an assistant condition")
        v = s[~na]
        if v.empty:
            continue
        if c.kind in ("int", "float"):
            num = pd.to_numeric(v, errors="coerce")
            if num.isna().any():
                errs.append(f"{table}.{c.name}: non-numeric values {list(v[num.isna()].unique()[:3])}")
                continue
            if c.kind == "int" and ((num % 1) != 0).any():
                errs.append(f"{table}.{c.name}: non-integer values")
            if c.lo is not None and (num < c.lo).any():
                errs.append(f"{table}.{c.name}: values below {c.lo}")
            if c.hi is not None and (num > c.hi).any():
                errs.append(f"{table}.{c.name}: values above {c.hi}")
        if c.allowed is not None:
            bad = ~v.astype(str).isin(c.allowed)
            if bad.any():
                errs.append(f"{table}.{c.name}: values not in {c.allowed}: {list(v[bad].unique()[:3])}")
        if c.pattern is not None:
            bad = ~v.astype(str).str.match(c.pattern)
            if bad.any():
                errs.append(f"{table}.{c.name}: values not matching {c.pattern}: {list(v[bad].unique()[:3])}")
    return errs


def empty_frame(table: str) -> pd.DataFrame:
    return pd.DataFrame(columns=[c.name for c in SCHEMAS[table]])


def read_table(path: Path, table: str) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"participant_id": str})
    errs = validate(df, table)
    if errs:
        raise SchemaError(f"{path}:\n  " + "\n  ".join(errs))
    return df


def schema_markdown() -> str:
    out = []
    for table, cols in SCHEMAS.items():
        out += [
            f"### `{table}.csv`",
            "",
            "| column | type | required | range / values | description |",
            "|---|---|---|---|---|",
        ]
        for c in cols:
            rng = ""
            if c.allowed:
                rng = ", ".join(c.allowed)
            elif c.lo is not None or c.hi is not None:
                lo = "" if c.lo is None else f"{c.lo:g}"
                hi = "" if c.hi is None else f"{c.hi:g}"
                rng = f">= {lo}" if not hi else f"{lo} – {hi}"
            if c.pattern:
                rng = f"`{c.pattern}`"
            req = "yes" if c.required else ("only for " + "/".join(c.na_unless) if c.na_unless else "no")
            out.append(f"| `{c.name}` | {c.kind} | {req} | {rng} | {c.desc} |")
        out.append("")
    return "\n".join(out)
