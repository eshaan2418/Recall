"""Rule-based query parser (on-device friendly; no LLM needed to route a question)."""

from __future__ import annotations

import re
from dataclasses import dataclass

VERB_ALIASES = {
    "take": "take",
    "took": "take",
    "pick up": "take",
    "picked up": "take",
    "grab": "take",
    "grabbed": "take",
    "use": "take",
    "used": "take",
    "put": "put",
    "put down": "put",
    "place": "put",
    "placed": "put",
    "leave": "put",
    "left": "put",
    "open": "open",
    "opened": "open",
    "close": "close",
    "closed": "close",
    "shut": "close",
    "turn on": "turn-on",
    "turned on": "turn-on",
    "switch on": "turn-on",
    "switched on": "turn-on",
    "turn off": "turn-off",
    "turned off": "turn-off",
    "switch off": "turn-off",
    "switched off": "turn-off",
    "wash": "wash",
    "washed": "wash",
    "rinse": "wash",
    "rinsed": "wash",
}
VERB_TO_CLASS = {"take": 0, "put": 1, "wash": 2, "open": 3, "close": 4, "turn-on": 6, "turn-off": 8}
_ART = r"(?:the |my |a |an |some )?"


@dataclass
class ParsedQuery:
    kind: str  # where | state | temporal
    obj: str
    verb: str | None = None


def _clean(s: str) -> str:
    s = s.strip().strip("?.! ").lower()
    return re.sub(r"\s+", " ", s)


def parse(question: str) -> ParsedQuery:
    q = _clean(question)
    m = re.match(rf"^(?:did|have) i (?:turn|switch)(?:ed)? off {_ART}(.+)$", q) or re.match(
        rf"^is {_ART}(.+?) (?:still )?(?:on|off)$", q
    )
    if m:
        return ParsedQuery("state", m.group(1))
    m = re.match(
        rf"^when did i (?:last )?(turn on|turn off|switch on|switch off|pick up|put down|\w+) {_ART}(.+?)(?: last)?$", q
    )
    if m:
        v = m.group(1).strip()
        verb = VERB_ALIASES.get(v)
        if verb is None:
            for alias, canon in sorted(VERB_ALIASES.items(), key=lambda kv: -len(kv[0])):
                if v.startswith(alias):
                    verb = canon
                    break
        return ParsedQuery("temporal", m.group(2), verb or "take")
    for pat in (
        rf"^where (?:is|are) {_ART}(.+)$",
        rf"^where did i (?:leave|put|place|last see|see|park) {_ART}(.+)$",
        r"^where did i (park)$",
        rf"^where'?s {_ART}(.+)$",
        rf"^have you seen {_ART}(.+)$",
        rf"^find {_ART}(.+)$",
    ):
        m = re.match(pat, q)
        if m:
            return ParsedQuery("where", m.group(1))
    return ParsedQuery("where", re.sub(rf"^{_ART}", "", q))
