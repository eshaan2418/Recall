"""M10 user-study package: counterbalancing, schema, and end-to-end analysis on SIMULATED data."""

from __future__ import annotations

import itertools
import math
from collections import Counter

import pandas as pd
import pytest

from recall.study import analysis, design, schema, simulate
from recall.study.simulate import SimParams


def _cfg(n: int = 12, seed: int = 7) -> design.StudyConfig:
    cfg = design.load_config()
    cfg.n_participants = n
    cfg.seed = seed
    return cfg


def test_williams_orders_are_all_permutations_and_carryover_balanced():
    orders = design.williams_orders(list(design.CONDITIONS))
    assert len(orders) == 6 and set(orders) == set(itertools.permutations(design.CONDITIONS))
    pairs = Counter((o[i], o[i + 1]) for o in orders for i in range(len(o) - 1))
    assert len(pairs) == 6 and set(pairs.values()) == {2}  # every ordered pair adjacent equally often


@pytest.mark.parametrize("n", [6, 12, 18])
def test_each_condition_equally_often_in_each_position(n):
    a = design.generate_assignments(_cfg(n))
    bal = design.balance_table(a)
    assert bal.shape == (3, 3)
    assert (bal.to_numpy() == n // 3).all()


def test_item_sets_balanced_and_assignments_valid():
    cfg = _cfg(12)
    a = design.generate_assignments(cfg)
    assert len(a) == 12 * 9
    # every item set paired with every condition equally often
    pairs = a.drop_duplicates(["participant_id", "condition"])
    assert (pd.crosstab(pairs["condition"], pairs["item_set"]).to_numpy() == 4).all()
    for _, g in a.groupby("participant_id"):
        assert sorted(g["item"]) == sorted(cfg.items)  # each item exactly once
        assert g["hiding_place"].is_unique
        assert sorted(g["trial_index"]) == list(range(1, 10))
        assert (g.groupby("block_index")["condition"].nunique() == 1).all()


def test_generator_is_seeded():
    a1 = design.generate_assignments(_cfg(12, seed=3))
    a2 = design.generate_assignments(_cfg(12, seed=3))
    a3 = design.generate_assignments(_cfg(12, seed=4))
    pd.testing.assert_frame_equal(a1, a2)
    assert not a1.equals(a3)


def test_schema_rejects_bad_rows():
    tabs = simulate.simulate_study(_cfg(6), seed=0)
    assert schema.validate(tabs["trials"], "trials") == []
    assert schema.validate(tabs["questionnaires"], "questionnaires") == []
    assert schema.validate(tabs["participants"], "participants") == []
    bad = tabs["trials"].copy()
    bad.loc[0, "success"] = 2
    bad.loc[1, "condition"] = "magic"
    errs = schema.validate(bad, "trials")
    assert any("success" in e for e in errs) and any("condition" in e for e in errs)
    q = tabs["questionnaires"].copy()
    q.loc[q["condition"] == "none", "sus_1"] = 3  # SUS must be NA for the no-assistant block
    assert any("sus_1" in e for e in schema.validate(q, "questionnaires"))


def test_holm():
    adj = analysis.holm({"a": 0.01, "b": 0.04, "c": 0.03})
    assert math.isclose(adj["a"], 0.03) and math.isclose(adj["c"], 0.06) and math.isclose(adj["b"], 0.06)


def test_analysis_end_to_end_recovers_planted_effect(tmp_path):
    params = SimParams(
        log_ratio={"none": 0.0, "b2": math.log(0.8), "recall": math.log(0.4)},
        sus_shift={"b2": 0.0, "recall": 15.0},
        tlx_shift={"none": 0.0, "b2": -5.0, "recall": -20.0},
    )
    tabs = simulate.simulate_study(_cfg(12), params, seed=11)
    data = tmp_path / "data"
    simulate.write_simulated(tabs, data)
    res = analysis.run(data, tmp_path / "out", simulated=True)
    c = res["time"]["contrasts"]
    assert c["recall_vs_b2"]["estimate"] < 0 and c["recall_vs_none"]["estimate"] < 0
    assert c["recall_vs_none"]["p_holm"] < 0.05
    assert res["questionnaires"]["tests"]["sus_score_recall_vs_b2"]["median_diff"] > 0
    assert res["questionnaires"]["tests"]["tlx_recall_vs_none"]["median_diff"] < 0
    report = (tmp_path / "out" / "report.md").read_text()
    assert report.count("SIMULATED") >= 2


def test_real_data_guard_and_results_guard(tmp_path):
    tabs = simulate.simulate_study(_cfg(6), seed=1)
    simulate.write_simulated(tabs, tmp_path)
    with pytest.raises(schema.SchemaError):
        analysis.run(tmp_path, tmp_path / "out", simulated=False)  # simulated rows can't pass as real
    with pytest.raises(ValueError):
        analysis.run(tmp_path, analysis.RESULTS / "m10_sim_should_fail", simulated=True)
    assert not (analysis.RESULTS / "m10_sim_should_fail").exists()


def test_missing_real_data_is_not_an_error(tmp_path, capsys):
    assert analysis.main(["--data-dir", str(tmp_path)]) == 0
    assert "No study data yet" in capsys.readouterr().out
