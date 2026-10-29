"""Metric arithmetic pinned on hand-computable prediction matrices."""

import pytest

from shotshift import build_report

LABELS = ["positive", "negative"]


def test_stable_matrix():
    # 2 queries x 2 orders, predictions never move
    preds = {0: {0: "positive", 1: "positive"}, 1: {0: "negative", 1: "negative"}}
    rep = build_report(LABELS, [(0, 1), (1, 0)], preds, answers=["positive", "positive"])
    assert rep.stability == 1.0
    assert rep.flip_rate == 0.0
    assert rep.fragile_rate == 0.0
    assert rep.canonical_accuracy == 0.5
    assert rep.mean_accuracy == 0.5
    assert rep.accuracy_range == (0.5, 0.5)
    assert rep.majority_vote_accuracy == 0.5
    assert rep.label_share == {"positive": 0.5, "negative": 0.5}
    assert rep.unparsed_rate == 0.0


def test_first_position_picker_matrix():
    # predictions follow whichever content sits first: they swap with the order
    preds = {0: {0: "positive", 1: "negative"}, 1: {0: "positive", 1: "negative"}}
    rep = build_report(LABELS, [(0, 1), (1, 0)], preds, answers=["positive", "negative"])
    for card in rep.cards:
        assert card.modal_share == 0.5
        assert card.n_flips == 1
    assert rep.stability == 0.5
    assert rep.flip_rate == 1.0
    assert rep.fragile_rate == 1.0
    # canonical order: both predicted positive -> 1/2 correct; swapped: both negative -> 1/2
    assert rep.accuracy_range == (0.5, 0.5)
    # modal label ties break by first appearance in order sequence -> canonical label
    assert rep.majority_vote_accuracy == 0.5


def test_accuracy_range_and_majority_vote():
    # query 0 correct everywhere; query 1 correct only at canonical order
    preds = {
        0: {0: "positive", 1: "positive", 2: "positive"},
        1: {0: "negative", 1: "positive", 2: "positive"},
    }
    rep = build_report(
        LABELS, [(0,), (1,), (2,)], preds, answers=["positive", "negative"]
    )
    accs = rep.accuracy_by_order()
    assert accs == [1.0, 0.5, 0.5]
    assert rep.canonical_accuracy == 1.0
    assert rep.mean_accuracy == pytest.approx(2 / 3)
    assert rep.accuracy_range == (0.5, 1.0)
    # modal of query 1 is positive (2 of 3) -> majority vote gets it wrong
    assert rep.majority_vote_accuracy == 0.5
    assert rep.cards[1].n_flips == 2
    assert rep.flip_rate == pytest.approx(2 / 4)


def test_unparsed_runs_are_excluded_not_counted_wrong():
    preds = {0: {0: "positive", 1: None}, 1: {0: None, 1: None}}
    rep = build_report(LABELS, [(0, 1), (1, 0)], preds, answers=["positive", "negative"])
    assert rep.unparsed_rate == 0.75
    assert rep.cards[0].modal_share == 1.0  # only parsed runs count
    assert rep.cards[1].modal is None
    assert rep.cards[1].modal_share == 0.0
    # accuracy at order 0 computed over parsed queries only (query 0)
    assert rep.accuracy_by_order()[0] == 1.0


def test_no_answers_no_accuracy_block():
    preds = {0: {0: "positive", 1: "negative"}}
    rep = build_report(LABELS, [(0, 1), (1, 0)], preds)
    assert rep.accuracy_by_order() is None
    assert rep.canonical_accuracy is None
    assert rep.majority_vote_accuracy is None
    assert "accuracy" not in rep.summary()


def test_validation():
    with pytest.raises(ValueError):
        build_report(["positive", "positive"], [(0,)], {0: {0: "positive"}})
    with pytest.raises(ValueError):
        build_report(LABELS, [(0,)], {0: {0: "maybe"}})
    with pytest.raises(ValueError):
        build_report(LABELS, [(0,)], {0: {0: "positive"}}, answers=["maybe"])
    with pytest.raises(ValueError):
        build_report(LABELS, [], {0: {0: "positive"}})


def test_json_round_trip_preserves_headline_metrics():
    preds = {
        0: {0: "positive", 1: "negative", 2: "positive"},
        1: {0: "negative", 1: "negative", 2: None},
    }
    rep = build_report(
        LABELS,
        [(0, 1), (1, 0), (0, 1)],
        preds,
        answers=["positive", "negative"],
        config={"mode": "score"},
    )
    from shotshift import ShotReport

    clone = ShotReport.from_dict(rep.to_dict())
    assert clone.stability == rep.stability
    assert clone.flip_rate == rep.flip_rate
    assert clone.accuracy_range == rep.accuracy_range
    assert clone.label_share == rep.label_share
    assert clone.to_dict()["metrics"] == rep.to_dict()["metrics"]
