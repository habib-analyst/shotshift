"""Core types and metric computation for ShotShift.

ShotShift measures how much a few-shot classifier's predictions move when
the order of its in-context examples changes. The experimental unit is a
*prompt* (one example set, in one order) applied to a set of queries.

Selections are labels (strings), never positions: a model whose prediction
depends on which example happens to sit first will select different labels
as the order rotates, and the report makes that movement maximal and
visible, which is the correct reading.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Optional, Sequence


def _validate_text(value: Any, what: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{what} must be a non-empty string, got {value!r}")
    return value


@dataclass(frozen=True)
class Example:
    """One in-context demonstration: an input text and its label."""

    text: str
    label: str

    def __post_init__(self) -> None:
        _validate_text(self.text, "example text")
        _validate_text(self.label, "example label")


@dataclass(frozen=True)
class Query:
    """One item to classify. ``answer`` is the gold label, if known."""

    text: str
    answer: Optional[str] = None

    def __post_init__(self) -> None:
        _validate_text(self.text, "query text")
        if self.answer is not None:
            _validate_text(self.answer, "query answer")


@dataclass
class QueryCard:
    """Per-query record across all probed orders.

    ``predictions`` maps order index -> predicted label (or ``None`` when a
    generate-mode response could not be parsed). ``canonical`` is the
    prediction at the identity order (order 0). ``modal`` is the most
    frequent non-None prediction; ``modal_share`` is its share of the
    parsed runs. ``n_flips`` counts non-canonical parsed runs whose
    prediction differs from the canonical prediction.
    """

    query_index: int
    predictions: dict[int, Optional[str]]
    canonical: Optional[str]
    modal: Optional[str]
    modal_share: float
    n_flips: int
    answer: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "query_index": self.query_index,
            "predictions": {str(k): v for k, v in sorted(self.predictions.items())},
            "canonical": self.canonical,
            "modal": self.modal,
            "modal_share": self.modal_share,
            "n_flips": self.n_flips,
            "answer": self.answer,
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "QueryCard":
        return cls(
            query_index=int(d["query_index"]),
            predictions={int(k): v for k, v in d["predictions"].items()},
            canonical=d["canonical"],
            modal=d["modal"],
            modal_share=float(d["modal_share"]),
            n_flips=int(d["n_flips"]),
            answer=d.get("answer"),
        )


def _modal(values: Sequence[str]) -> tuple[Optional[str], float]:
    """Most frequent value and its share. Ties break toward the value that
    appears first in ``values`` (deterministic, order-of-appearance)."""
    if not values:
        return None, 0.0
    counts: dict[str, int] = {}
    for v in values:
        counts[v] = counts.get(v, 0) + 1
    best = max(values, key=lambda v: counts[v])  # first max in appearance order
    return best, counts[best] / len(values)


@dataclass
class ShotReport:
    """Aggregate result of one probe run.

    Headline metrics:

    - ``stability``: mean over queries of the modal share — how often the
      most-chosen label was chosen, across orders. 1.0 means no prediction
      ever moved.
    - ``flip_rate``: share of non-canonical parsed (order, query) runs
      whose prediction differs from that query's canonical-order
      prediction.
    - ``fragile_rate``: share of queries whose prediction moves under at
      least one order.
    - Accuracy block (only when every query carries a gold label):
      accuracy at the canonical order, mean/min/max over orders, the
      accuracy range (max - min: the part of a single reported accuracy
      that is order luck), and majority-vote accuracy (scoring the modal
      label instead of the canonical-order prediction).
    - ``label_share``: mean predicted-label distribution over orders.
    """

    labels: list[str]
    orders: list[tuple[int, ...]]
    cards: list[QueryCard]
    config: dict[str, Any] = field(default_factory=dict)

    # ---- derived metrics -------------------------------------------------
    @property
    def n_orders(self) -> int:
        return len(self.orders)

    @property
    def n_queries(self) -> int:
        return len(self.cards)

    @property
    def stability(self) -> float:
        if not self.cards:
            return 0.0
        return sum(c.modal_share for c in self.cards) / len(self.cards)

    @property
    def flip_rate(self) -> float:
        flips = 0
        total = 0
        for c in self.cards:
            if c.canonical is None:
                continue
            for oi, pred in c.predictions.items():
                if oi == 0 or pred is None:
                    continue
                total += 1
                if pred != c.canonical:
                    flips += 1
        return flips / total if total else 0.0

    @property
    def fragile_rate(self) -> float:
        if not self.cards:
            return 0.0
        fragile = sum(1 for c in self.cards if c.n_flips > 0)
        return fragile / len(self.cards)

    @property
    def unparsed_rate(self) -> float:
        total = sum(len(c.predictions) for c in self.cards)
        if not total:
            return 0.0
        unparsed = sum(
            1 for c in self.cards for p in c.predictions.values() if p is None
        )
        return unparsed / total

    @property
    def _has_answers(self) -> bool:
        return bool(self.cards) and all(c.answer is not None for c in self.cards)

    def accuracy_by_order(self) -> Optional[list[float]]:
        if not self._has_answers:
            return None
        out: list[float] = []
        for oi in range(self.n_orders):
            parsed = [
                c for c in self.cards if c.predictions.get(oi) is not None
            ]
            if not parsed:
                out.append(0.0)
                continue
            hits = sum(1 for c in parsed if c.predictions[oi] == c.answer)
            out.append(hits / len(parsed))
        return out

    @property
    def canonical_accuracy(self) -> Optional[float]:
        accs = self.accuracy_by_order()
        return accs[0] if accs else None

    @property
    def mean_accuracy(self) -> Optional[float]:
        accs = self.accuracy_by_order()
        return sum(accs) / len(accs) if accs else None

    @property
    def accuracy_range(self) -> Optional[tuple[float, float]]:
        accs = self.accuracy_by_order()
        return (min(accs), max(accs)) if accs else None

    @property
    def majority_vote_accuracy(self) -> Optional[float]:
        if not self._has_answers:
            return None
        parsed = [c for c in self.cards if c.modal is not None]
        if not parsed:
            return 0.0
        return sum(1 for c in parsed if c.modal == c.answer) / len(parsed)

    @property
    def label_share(self) -> dict[str, float]:
        counts: dict[str, int] = {lab: 0 for lab in self.labels}
        total = 0
        for c in self.cards:
            for pred in c.predictions.values():
                if pred is not None:
                    counts[pred] = counts.get(pred, 0) + 1
                    total += 1
        if not total:
            return {lab: 0.0 for lab in self.labels}
        return {lab: counts.get(lab, 0) / total for lab in self.labels}

    # ---- reporting --------------------------------------------------------
    def summary(self) -> str:
        lines = [
            f"ShotShift report: {self.n_queries} quer"
            f"{'y' if self.n_queries == 1 else 'ies'}, "
            f"{self.n_orders} order(s) of {len(self.orders[0]) if self.orders else 0} example(s), "
            f"stability {self.stability:.3f}, flip rate {self.flip_rate:.3f}, "
            f"fragile queries {self.fragile_rate:.3f}"
        ]
        if self.unparsed_rate:
            lines.append(f"  unparsed rate {self.unparsed_rate:.3f}")
        accs = self.accuracy_by_order()
        if accs is not None:
            lo, hi = min(accs), max(accs)
            lines.append(
                f"  canonical accuracy {accs[0]:.3f}; mean over orders "
                f"{sum(accs) / len(accs):.3f}; range [{lo:.3f}, {hi:.3f}] "
                f"(width {hi - lo:.3f}); majority-vote accuracy "
                f"{self.majority_vote_accuracy:.3f}"
            )
        shares = ", ".join(
            f"{lab} {share:.3f}" for lab, share in self.label_share.items()
        )
        lines.append(f"  predicted-label share: {shares}")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "labels": list(self.labels),
            "orders": [list(o) for o in self.orders],
            "cards": [c.to_dict() for c in self.cards],
            "config": self.config,
            "metrics": {
                "stability": self.stability,
                "flip_rate": self.flip_rate,
                "fragile_rate": self.fragile_rate,
                "unparsed_rate": self.unparsed_rate,
                "canonical_accuracy": self.canonical_accuracy,
                "mean_accuracy": self.mean_accuracy,
                "accuracy_range": (
                    list(self.accuracy_range) if self.accuracy_range else None
                ),
                "majority_vote_accuracy": self.majority_vote_accuracy,
                "label_share": self.label_share,
            },
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "ShotReport":
        return cls(
            labels=list(d["labels"]),
            orders=[tuple(o) for o in d["orders"]],
            cards=[QueryCard.from_dict(c) for c in d["cards"]],
            config=dict(d.get("config", {})),
        )


def build_report(
    labels: Sequence[str],
    orders: Sequence[Sequence[int]],
    predictions: Mapping[int, Mapping[int, Optional[str]]],
    answers: Optional[Sequence[Optional[str]]] = None,
    config: Optional[dict[str, Any]] = None,
) -> ShotReport:
    """Assemble a ShotReport from raw per-(query, order) predictions.

    ``predictions[query_index][order_index]`` is the predicted label or
    ``None`` (unparsed). Every prediction must be one of ``labels``.
    """
    label_set = set(labels)
    if not label_set or len(label_set) != len(labels):
        raise ValueError("labels must be a non-empty list of distinct strings")
    for lab in labels:
        _validate_text(lab, "label")
    order_list = [tuple(o) for o in orders]
    if not order_list:
        raise ValueError("at least one order is required")
    cards: list[QueryCard] = []
    for qi in sorted(predictions):
        preds = predictions[qi]
        cleaned: dict[int, Optional[str]] = {}
        for oi, pred in preds.items():
            if pred is not None and pred not in label_set:
                raise ValueError(
                    f"prediction {pred!r} for query {qi} is not one of the labels"
                )
            cleaned[int(oi)] = pred
        canonical = cleaned.get(0)
        parsed_in_order = [
            cleaned[oi] for oi in sorted(cleaned) if cleaned[oi] is not None
        ]
        modal, share = _modal(parsed_in_order)
        n_flips = 0
        if canonical is not None:
            n_flips = sum(
                1
                for oi, p in cleaned.items()
                if oi != 0 and p is not None and p != canonical
            )
        answer = answers[qi] if answers is not None else None
        if answer is not None and answer not in label_set:
            raise ValueError(f"gold answer {answer!r} is not one of the labels")
        cards.append(
            QueryCard(
                query_index=qi,
                predictions=cleaned,
                canonical=canonical,
                modal=modal,
                modal_share=share,
                n_flips=n_flips,
                answer=answer,
            )
        )
    if not cards:
        raise ValueError("at least one query is required")
    return ShotReport(
        labels=list(labels),
        orders=order_list,
        cards=cards,
        config=config or {},
    )


def check_finite_scores(scores: Mapping[str, float], labels: Iterable[str]) -> None:
    """Validate a score-mode callable result against the label set."""
    label_set = set(labels)
    keys = set(scores)
    missing = label_set - keys
    extra = keys - label_set
    if missing:
        raise ValueError(f"score mapping is missing labels: {sorted(missing)}")
    if extra:
        raise ValueError(f"score mapping has unknown labels: {sorted(extra)}")
    for lab, value in scores.items():
        if not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"score for label {lab!r} is not finite: {value!r}")
