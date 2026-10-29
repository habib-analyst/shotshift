"""Probe drivers against scripted scorers with by-construction behaviour."""

import re

import pytest

from shotshift import Example, Query, leave_one_out, probe, probe_zero_shot

POS = Example("a glowing review", "positive")
NEG = Example("a dreadful review", "negative")
QUERIES = [Query("some film", answer="positive"), Query("other film", answer="negative")]

_LABEL_RE = re.compile(r"Label: (\w+)")


def first_demo_scorer(prompt: str) -> dict[str, float]:
    """Predicts the label of whichever example sits first in the prompt."""
    first = _LABEL_RE.findall(prompt)[0]
    return {"positive": 1.0 if first == "positive" else 0.0,
            "negative": 1.0 if first == "negative" else 0.0}


def last_demo_scorer(prompt: str) -> dict[str, float]:
    """Predicts the label of whichever example sits last (pure recency)."""
    last = _LABEL_RE.findall(prompt)[-1]
    return {"positive": 1.0 if last == "positive" else 0.0,
            "negative": 1.0 if last == "negative" else 0.0}


def constant_scorer(prompt: str) -> dict[str, float]:
    return {"positive": 2.0, "negative": 1.0}


def test_first_demo_scorer_is_maximally_order_sensitive():
    rep = probe(first_demo_scorer, [POS, NEG], QUERIES)
    assert rep.n_orders == 2  # 2! exhaustive
    assert rep.config["exhaustive"] is True
    assert rep.stability == 0.5
    assert rep.flip_rate == 1.0
    assert rep.fragile_rate == 1.0


def test_constant_scorer_is_perfectly_stable():
    rep = probe(constant_scorer, [POS, NEG], QUERIES)
    assert rep.stability == 1.0
    assert rep.flip_rate == 0.0
    assert rep.label_share == {"positive": 1.0, "negative": 0.0}
    assert rep.canonical_accuracy == 0.5  # stable and half wrong: both numbers print


def test_generate_mode_parses_and_counts_unparsed():
    def gen(prompt: str) -> str:
        return f"Label: {_LABEL_RE.findall(prompt)[0]}"

    rep = probe(gen, [POS, NEG], QUERIES, mode="generate")
    assert rep.unparsed_rate == 0.0
    assert rep.stability == 0.5

    def garbage(prompt: str) -> str:
        return "I cannot decide about this one at all, truly."

    rep2 = probe(garbage, [POS, NEG], QUERIES, mode="generate")
    assert rep2.unparsed_rate == 1.0
    assert rep2.cards[0].modal is None


def test_zero_shot_probe():
    rep = probe_zero_shot(constant_scorer, ["positive", "negative"], QUERIES)
    assert rep.n_orders == 1
    assert rep.canonical_accuracy == 0.5
    assert rep.stability == 1.0


def test_leave_one_out_recency_scorer():
    result = leave_one_out(last_demo_scorer, [POS, NEG], QUERIES)
    entries = {e["removed_index"]: e for e in result["leave_one_out"]}
    # full set: identity order predicts negative (last demo), swapped predicts
    # positive -> mean accuracy 0.5 against golds [positive, negative] averaged
    # per query... per-order accuracies are [0.5, 0.5] (one query right each time)
    assert result["full"]["metrics"]["mean_accuracy"] == 0.5
    # removing POS leaves only NEG: always predicts negative -> queries split
    # [wrong, right] at every order -> mean accuracy still 0.5, delta 0.0;
    # stability rises to 1.0 (single example cannot move)
    assert entries[0]["delta_mean_accuracy"] == 0.0
    assert entries[0]["delta_stability"] == 0.5
    assert entries[1]["delta_mean_accuracy"] == 0.0
    assert entries[1]["delta_stability"] == 0.5


def test_probe_validation():
    with pytest.raises(ValueError):
        probe(constant_scorer, [POS, NEG], QUERIES, mode="bogus")
    with pytest.raises(ValueError):
        probe(constant_scorer, [POS, NEG], [])
    with pytest.raises(ValueError):
        probe(constant_scorer, [], QUERIES)
    with pytest.raises(ValueError):
        probe(lambda p: {"positive": 1.0}, [POS, NEG], QUERIES)  # missing label
    with pytest.raises(ValueError):
        probe(lambda p: {"positive": float("nan"), "negative": 0.0}, [POS, NEG], QUERIES)
    with pytest.raises(ValueError):
        probe(lambda p: "positive", [POS, NEG], QUERIES)  # wrong return type
    with pytest.raises(ValueError):
        probe(constant_scorer, [POS, NEG], [Query("x", answer="neutral")])
    with pytest.raises(ValueError):
        leave_one_out(constant_scorer, [POS], QUERIES)


def test_report_json_round_trip_through_probe():
    from shotshift import ShotReport

    rep = probe(first_demo_scorer, [POS, NEG], QUERIES, seed=3)
    clone = ShotReport.from_dict(rep.to_dict())
    assert clone.to_dict() == rep.to_dict()
    # determinism: same seed, same report
    rep2 = probe(first_demo_scorer, [POS, NEG], QUERIES, seed=3)
    assert rep2.to_dict() == rep.to_dict()
