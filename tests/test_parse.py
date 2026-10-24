import pytest

from recall.qa.parse import parse


@pytest.mark.parametrize(
    "q,kind,obj,verb",
    [
        ("Where is the knife?", "where", "knife", None),
        ("Where did I leave my keys?", "where", "keys", None),
        ("where's the chopping board", "where", "chopping board", None),
        ("Did I turn off the hob?", "state", "hob", None),
        ("Is the tap still on?", "state", "tap", None),
        ("When did I last open the fridge?", "temporal", "fridge", "open"),
        ("When did I last turn off the tap?", "temporal", "tap", "turn-off"),
        ("When did I last take my pills?", "temporal", "pills", "take"),
        ("Where did I park?", "where", "park", None),
    ],
)
def test_parse(q, kind, obj, verb):
    p = parse(q)
    assert p.kind == kind
    assert p.obj == obj
    if verb:
        assert p.verb == verb
