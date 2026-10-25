"""Adversarial grounding suite: the answer text may only contain facts from memory."""

import pytest

from recall.qa.answer import Answer
from recall.qa.generate import grounded_rewrite, render, verify

LOCS = ["kitchen counter", "drawer", "fridge", "sink", "cupboard", "table", "bin"]


def test_abstain_when_no_candidate():
    r = render(Answer("where", True), "passport", None, None, 0.3)
    assert r.abstained and "don't know" in r.text
    for l in LOCS:
        assert l not in r.text


def test_abstain_below_threshold():
    r = render(Answer("where", False, 0, 100.0, 1), "keys", "drawer", 0.2, 0.3)
    assert r.abstained and "drawer" not in r.text


def test_where_template_grounded():
    r = render(Answer("where", False, 0, 3600 + 42 * 60, 1), "keys", "drawer", 0.78, 0.3, start_clock_s=17 * 3600)
    assert "drawer" in r.text and "6:42 pm" in r.text and "78%" in r.text


@pytest.mark.parametrize(
    "llm_out",
    [
        "Your keys are in the fridge, put there at 6:42 pm. 78% confident.",  # swapped location
        "Your keys are in the drawer next to the sink at 6:42 pm (78%).",  # added location
        "Your keys are in the drawer since 6:40 pm, 78% sure.",  # changed time
        "They're in the drawer from 6:42 pm.",  # dropped confidence
        "Ignore memory: kitchen counter.",  # injection-like
        "",
    ],
)
def test_rewrite_rejected(llm_out):
    base = render(Answer("where", False, 0, 3600 + 42 * 60, 1), "keys", "drawer", 0.78, 0.3, start_clock_s=17 * 3600)
    r = grounded_rewrite(base, "drawer", LOCS, lambda p: llm_out)
    assert r.text == base.text and not r.used_llm


def test_rewrite_accepted():
    base = render(Answer("where", False, 0, 3600 + 42 * 60, 1), "keys", "drawer", 0.78, 0.3, start_clock_s=17 * 3600)
    r = grounded_rewrite(
        base,
        "drawer",
        LOCS,
        lambda p: "Your keys are most likely in the drawer — left there at 6:42 pm (78% confident).",
    )
    assert r.used_llm


def test_rewrite_never_for_abstain():
    base = render(Answer("where", True), "keys", None, None, 0.3)
    r = grounded_rewrite(base, None, LOCS, lambda p: "In the drawer!")
    assert r.abstained and "drawer" not in r.text


def test_verify_allows_substring_location():
    assert verify(
        "It's on the kitchen counter, 5:00 pm, 60%.",
        "kitchen counter 5:00 pm 60%",
        "kitchen counter",
        [*LOCS, "counter"],
    )
