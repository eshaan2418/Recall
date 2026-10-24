"""Natural-language answers grounded in memory (never invents a location).

Pipeline: structured :class:`Answer` + calibrated confidence → **template** sentence whose slot values
(location, time, object) come only from the stored event. An optional small on-device LLM may *rephrase*
the template; its output is accepted only if it (a) contains the location string and clock time verbatim,
(b) mentions no other location from the vocabulary, (c) keeps the confidence number. Otherwise the
template is returned. Abstentions are never rephrased.

This makes "location hallucination" impossible by construction; ``tests/test_generate.py`` checks it on
an adversarial suite.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass

from recall.qa.answer import Answer

ABSTAIN_TEXT = "I don't know — I haven't seen the {obj} in my memory{since}."
LOW_CONF_TEXT = "I'm not sure. My best guess is the {loc}, last seen at {time} — but only {conf}% confident."


@dataclass
class Rendered:
    text: str
    abstained: bool
    used_llm: bool
    evidence_t: float | None
    confidence: float | None


def clock(t_s: float, start_clock_s: float = 0.0) -> str:
    s = int(start_clock_s + t_s) % 86400
    h, m = divmod(s // 60, 60)
    ampm = "am" if h < 12 else "pm"
    return f"{(h % 12) or 12}:{m:02d} {ampm}"


def render(
    ans: Answer,
    obj: str,
    loc_name: str | None,
    conf: float | None,
    abstain_below: float,
    start_clock_s: float = 0.0,
    state_name: str | None = None,
) -> Rendered:
    """Deterministic, grounded template."""
    if ans.abstain or conf is None or conf < abstain_below:
        return Rendered(ABSTAIN_TEXT.format(obj=obj, since=""), True, False, None, conf)
    pct = int(round(100 * conf))
    tm = clock(ans.t or 0.0, start_clock_s)
    if ans.kind == "where":
        assert loc_name is not None
        if conf < 0.5:
            return Rendered(LOW_CONF_TEXT.format(loc=loc_name, time=tm, conf=pct), False, False, ans.t, conf)
        txt = f"Probably the {loc_name} — you put the {obj} there at {tm}. {pct}% confident."
    elif ans.kind == "temporal":
        txt = f"Around {tm}. {pct}% confident."
    else:
        st = state_name or ans.state or "unknown"
        txt = f"The last time I saw you at the {obj} was {tm}; it looked {st} after that. {pct}% confident."
    return Rendered(txt, False, False, ans.t, conf)


def grounded_rewrite(
    r: Rendered, loc_name: str | None, all_locations: list[str], llm: Callable[[str], str] | None
) -> Rendered:
    """Optionally let an LLM rephrase; reject any output that changes or adds facts."""
    if llm is None or r.abstained:
        return r
    prompt = (
        "Rephrase this answer for a friendly voice assistant in one short sentence. Keep every place name, "
        f"time and percentage exactly as written. Do not add information.\nAnswer: {r.text}\nRephrased:"
    )
    out = llm(prompt).strip().split("\n")[0]
    if not verify(out, r.text, loc_name, all_locations):
        return r
    return Rendered(out, False, True, r.evidence_t, r.confidence)


def verify(candidate: str, template: str, loc_name: str | None, all_locations: list[str]) -> bool:
    c = candidate.lower()
    if not c or len(c) > 3 * len(template) + 40:
        return False
    if loc_name is not None and loc_name.lower() not in c:
        return False
    for t in re.findall(r"\d{1,2}:\d{2} [ap]m", template):
        if t not in c:
            return False
    for p in re.findall(r"\d+%", template):
        if p not in c:
            return False
    others = [l for l in all_locations if l.lower() != (loc_name or "").lower()]
    for l in others:
        if loc_name and l.lower() in loc_name.lower():
            continue  # e.g. "counter" inside "kitchen counter"
        if re.search(rf"\b{re.escape(l.lower())}\b", c):
            return False
    return True
