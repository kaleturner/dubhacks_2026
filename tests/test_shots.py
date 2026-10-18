import pytest

from shared.constants import SHOTS
from shared.schemas import Intent
from shared.shots import disabled_note, fallback_shots, guess_object, normalize_shots, parse_total

ALL = {name: {**spec, "enabled": True, "min_s": 1} for name, spec in SHOTS.items()}


def total(shots):
    return round(sum(s["duration_s"] for s in shots), 2)


def test_phase_one_only_zoom():
    shots = fallback_shots("make a product ad for this apple")
    assert [s["task"] for s in shots] == ["zoom"]
    assert total(shots) == 15


def test_disabled_move_is_reported_not_swapped_silently():
    text = "20 second overhead shot of this apple"
    assert [s["task"] for s in fallback_shots(text)] == ["zoom"]
    assert "overhead" in disabled_note(text)
    assert disabled_note("zoom in on the apple") is None


def test_named_moves_keep_order_and_total():
    shots = fallback_shots("20 second overhead then zoom", registry=ALL)
    assert [s["task"] for s in shots] == ["overhead", "zoom"]
    assert total(shots) == 20


def test_vague_query_uses_story_order():
    shots = fallback_shots("make a product ad", registry=ALL)
    assert [s["task"] for s in shots] == ["sweep", "overhead", "zoom"]
    assert total(shots) == 15


def test_normalize_drops_unknown_and_caps():
    raw = [{"task": "orbit", "duration_s": 5}] + [{"task": "zoom", "duration_s": 5}] * 5
    shots = normalize_shots(raw, 12, ALL)
    assert len(shots) == 3
    assert all(s["task"] == "zoom" for s in shots)
    assert total(shots) == 12


def test_normalize_empty_and_garbage():
    assert normalize_shots([], 10) == []
    assert normalize_shots(None, 10) == []
    assert normalize_shots([{"task": "sweep", "duration_s": 5}], 10) == []


def test_normalize_clamps_total_and_single_max():
    assert total(normalize_shots([{"task": "zoom", "duration_s": 5}], 99)) == 30
    assert total(normalize_shots([{"task": "zoom", "duration_s": 5}], 0)) == 4


def test_normalize_respects_min_by_dropping_tail():
    shots = normalize_shots([{"task": "zoom"}, {"task": "sweep"}], 5)
    assert [s["task"] for s in shots] == ["zoom"]


def test_helpers():
    assert parse_total("record 12 seconds of this") == 12
    assert parse_total("record this") == 15
    assert guess_object("record a 15 second video of this apple") == "apple"
    assert guess_object("go") == "apple"


def test_intent_validation():
    intent = Intent(object="apple", style="x", shots=[{"task": "zoom", "duration_s": 15}])
    assert intent.duration_s == 15
    with pytest.raises(ValueError):
        Intent(object="a", style="x", shots=[{"task": "zoom", "duration_s": 20}, {"task": "zoom", "duration_s": 20}])
