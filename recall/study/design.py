"""M10 randomization / counterbalancing for the within-subjects user study.

Design
------
* 3 retrieval conditions: ``recall`` (Recall: answer + confidence + evidence thumbnail), ``none`` (no assistant,
  memory only) and ``b2`` (frame-sample + CLIP retrieval: best-matching frame, no confidence).
* Condition ORDER is counterbalanced with a Williams design. For an odd number of conditions this needs two
  Latin squares (here: all 3! = 6 orders), which balances position AND first-order carry-over. Complete blocks
  of 6 participants are exactly balanced; n = 12 (or 6, 18) is recommended.
* Item SETS (items split into 3 fixed sets) are rotated over conditions with a second Latin square, so every
  item set is paired with every condition equally often (n multiple of 3) and the rotation is not confounded
  with the order (it is shifted every 6 participants).
* Design rows are assigned to pseudonymous participant IDs by a seeded permutation within each block of 6;
  hiding places, put-down (encoding) order and within-block retrieval order are seeded per participant.

A replacement for a participant who withdraws takes the SAME design row (same order, same item sets) so the
balance is preserved.

Usage: ``python -m recall.study.design [--config study/study_config.yaml] [--out study/assignments]``
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from recall.paths import REPO

STUDY_DIR = REPO / "study"
DEFAULT_CONFIG = STUDY_DIR / "study_config.yaml"
CONDITIONS = ("recall", "none", "b2")
CONDITION_LABELS = {
    "recall": "Recall (answer + confidence + evidence thumbnail)",
    "none": "No assistant (memory only)",
    "b2": "Baseline B2 (frame-sample + CLIP best frame, no confidence)",
}


@dataclass
class StudyConfig:
    seed: int = 20261025
    n_participants: int = 12
    items_per_condition: int = 3
    time_cap_s: float = 180.0
    retention_delay_min: int = 30
    conditions: list[str] = field(default_factory=lambda: list(CONDITIONS))
    items: list[str] = field(default_factory=list)
    hiding_places: list[str] = field(default_factory=list)
    filler_tasks: list[str] = field(default_factory=list)

    def validate(self) -> None:
        k = len(self.conditions)
        if sorted(self.conditions) != sorted(CONDITIONS):
            raise ValueError(f"conditions must be a permutation of {CONDITIONS}")
        if len(self.items) != k * self.items_per_condition:
            raise ValueError(f"need exactly {k * self.items_per_condition} items, got {len(self.items)}")
        if len(set(self.items)) != len(self.items):
            raise ValueError("items must be unique")
        if len(set(self.hiding_places)) < len(self.items):
            raise ValueError("need at least as many distinct hiding places as items")
        if not 1 <= self.n_participants <= 99:
            raise ValueError("n_participants must be in 1..99")


def default_config() -> StudyConfig:
    return StudyConfig(
        items=["keys", "wallet", "sunglasses", "tv_remote", "earbuds_case", "usb_stick", "pen", "lip_balm", "watch"],
        hiding_places=[f"place_{i:02d}" for i in range(1, 13)],
        filler_tasks=["pour a glass of water", "fold the towel", "wipe the counter"],
    )


def load_config(path: Path | None = None) -> StudyConfig:
    path = path or DEFAULT_CONFIG
    if not path.exists():
        return default_config()
    raw = yaml.safe_load(path.read_text()) or {}
    cfg = StudyConfig(**{k: v for k, v in raw.items() if k in StudyConfig.__dataclass_fields__})
    cfg.validate()
    return cfg


def williams_orders(conditions: tuple[str, ...] | list[str]) -> list[tuple[str, ...]]:
    """Williams (carry-over balanced) Latin square rows. Odd k -> 2k rows (square + its mirror)."""
    k = len(conditions)
    # standard construction: first row 0, 1, k-1, 2, k-2, ...; row r adds r mod k
    first = [0]
    lo, hi = 1, k - 1
    while len(first) < k:
        first.append(lo)
        lo += 1
        if len(first) < k:
            first.append(hi)
            hi -= 1
    rows = [[(x + r) % k for x in first] for r in range(k)]
    if k % 2 == 1:
        rows += [list(reversed(r)) for r in rows]
    return [tuple(conditions[i] for i in r) for r in rows]


def participant_id(i: int) -> str:
    return f"P{i + 1:02d}"


def design_rows(n: int, conditions: list[str], seed: int) -> list[dict]:
    """Return one design row per participant (order + item-set rotation), balanced in complete blocks."""
    orders = williams_orders(conditions)
    m = len(orders)
    k = len(conditions)
    rng = np.random.default_rng(seed)
    rows: list[dict] = []
    for block_start in range(0, n, m):
        block = []
        for j in range(m):
            p = block_start + j
            rotation = (j + block_start // m) % k  # shift the set rotation every block -> not confounded with order
            block.append({"design_row": p, "order": orders[j], "set_rotation": rotation})
        # seeded allocation of design rows to the participants of this block
        perm = rng.permutation(m)
        block = [block[i] for i in perm]
        rows.extend(block)
    return rows[:n]


def item_sets(items: list[str], k: int) -> list[list[str]]:
    per = len(items) // k
    return [items[i * per : (i + 1) * per] for i in range(k)]


def generate_assignments(cfg: StudyConfig) -> pd.DataFrame:
    """Long table: one row per participant x retrieval trial."""
    cfg.validate()
    k = len(cfg.conditions)
    sets = item_sets(cfg.items, k)
    canon = list(CONDITIONS)
    rows = design_rows(cfg.n_participants, cfg.conditions, cfg.seed)
    out = []
    for i, dr in enumerate(rows):
        pid = participant_id(i)
        prng = np.random.default_rng([cfg.seed, i])
        places = list(prng.choice(cfg.hiding_places, size=len(cfg.items), replace=False))
        place_of = dict(zip(cfg.items, places))
        encode_order = list(prng.permutation(cfg.items))
        cond_set = {c: (canon.index(c) + dr["set_rotation"]) % k for c in canon}
        t = 0
        for b, cond in enumerate(dr["order"]):
            block_items = list(prng.permutation(sets[cond_set[cond]]))
            for it in block_items:
                t += 1
                out.append(
                    {
                        "participant_id": pid,
                        "design_row": dr["design_row"],
                        "order_label": "-".join(dr["order"]),
                        "block_index": b + 1,
                        "condition": cond,
                        "trial_index": t,
                        "item": it,
                        "item_set": cond_set[cond] + 1,
                        "hiding_place": place_of[it],
                        "encoding_step": encode_order.index(it) + 1,
                    }
                )
    return pd.DataFrame(out)


def balance_table(assign: pd.DataFrame) -> pd.DataFrame:
    """Count of participants with each condition in each block position."""
    first = assign.drop_duplicates(["participant_id", "block_index"])
    return pd.crosstab(first["condition"], first["block_index"])


def run_sheet(pid: str, a: pd.DataFrame, cfg: StudyConfig) -> str:
    """Experimenter sheet for one participant (encoding script + retrieval trial list)."""
    a = a.sort_values("trial_index")
    enc = a.sort_values("encoding_step")
    fillers = cfg.filler_tasks or ["(no filler task)"]
    lines = [
        f"# Run sheet — {pid}  (design row {int(a['design_row'].iloc[0])}, order {a['order_label'].iloc[0]})",
        "",
        f"Generated by `recall.study.design` with seed {cfg.seed}. Do not edit; regenerate from the config.",
        "",
        "## Phase 1 — Encoding walk (camera recording ON)",
        "",
        "Hand the participant one card at a time. Put-downs (P) are interleaved with filler tasks (F).",
        "",
        "| step | card |",
        "|---|---|",
    ]
    step = 0
    for j, (_, r) in enumerate(enc.iterrows()):
        step += 1
        lines.append(f"| {step} | P: put the **{r['item']}** in/at **{r['hiding_place']}** |")
        if j % 2 == 1 or j == len(enc) - 1:
            step += 1
            lines.append(f"| {step} | F: {fillers[(j // 2) % len(fillers)]} |")
    lines += [
        "",
        f"## Phase 2 — Retention delay (>= {cfg.retention_delay_min} min; camera OFF)",
        "",
        "## Phase 3 — Retrieval trials",
        "",
        f"Time cap per trial: {cfg.time_cap_s:.0f} s. Record each row in `trials.csv`; after each block, the",
        "block questionnaire (NASA-TLX; SUS + trust/usefulness for recall and b2) goes in `questionnaires.csv`.",
        "",
        "| trial | block | condition | item | correct place (experimenter only) | time (s) | found (y/n) |",
        "|---|---|---|---|---|---|---|",
    ]
    for _, r in a.iterrows():
        lines.append(
            f"| {r['trial_index']} | {r['block_index']} | {r['condition']} | {r['item']} | {r['hiding_place']} | | |"
        )
    return "\n".join(lines) + "\n"


def write_assignments(cfg: StudyConfig, out_dir: Path) -> pd.DataFrame:
    out_dir.mkdir(parents=True, exist_ok=True)
    a = generate_assignments(cfg)
    a.to_csv(out_dir / "assignments.csv", index=False)
    sheets = out_dir / "run_sheets"
    sheets.mkdir(exist_ok=True)
    for pid, g in a.groupby("participant_id"):
        (sheets / f"{pid}.md").write_text(run_sheet(str(pid), g, cfg))
    bal = balance_table(a)
    note = ""
    m = len(williams_orders(cfg.conditions))
    if cfg.n_participants % m:
        note = (
            f"\nWARNING: n={cfg.n_participants} is not a multiple of {m}; the last block is incomplete, so "
            "positions are only approximately balanced. Prefer n = 6, 12 or 18.\n"
        )
    (out_dir / "balance.txt").write_text(
        f"seed={cfg.seed} n={cfg.n_participants}\nparticipants per condition x block position:\n{bal.to_string()}\n"
        + note
    )
    return a


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    ap.add_argument("--out", type=Path, default=STUDY_DIR / "assignments")
    ap.add_argument("--n", type=int, default=None, help="override n_participants")
    ap.add_argument("--seed", type=int, default=None, help="override seed")
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    if args.n is not None:
        cfg.n_participants = args.n
    if args.seed is not None:
        cfg.seed = args.seed
    a = write_assignments(cfg, args.out)
    print((args.out / "balance.txt").read_text())
    print(f"wrote {len(a)} trial rows for {cfg.n_participants} participants -> {args.out}")


if __name__ == "__main__":
    main()
