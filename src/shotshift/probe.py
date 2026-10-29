"""Probe drivers: run a model callable over orders x queries.

Two protocols, mirroring the two ways few-shot classifiers are actually
run:

- **Score mode** (default): the callable receives a rendered prompt and
  returns a mapping of label -> score (typically the logprob of the label
  continuation). The prediction is the argmax label; ties break toward
  the label that appears first in the label set.
- **Generate mode**: the callable returns generated text, and a
  conservative parser reads the label out of it. Anything unparseable is
  counted in the report's unparsed rate, never guessed.

The label set is the sorted-by-first-appearance union of the examples'
labels. Gold answers, when supplied, must belong to it.
"""

from __future__ import annotations

from typing import Any, Callable, Optional, Sequence

from .core import Example, Query, ShotReport, build_report, check_finite_scores
from .parsing import parse_label
from .prompts import (
    DEFAULT_EXAMPLE_TEMPLATE,
    DEFAULT_INSTRUCTION,
    DEFAULT_QUERY_TEMPLATE,
    DEFAULT_SEPARATOR,
    describe_templates,
    make_orders,
    render_prompt,
)

ScoreFn = Callable[[str], dict[str, float]]
GenerateFn = Callable[[str], str]


def _label_set(examples: Sequence[Example]) -> list[str]:
    labels: list[str] = []
    for ex in examples:
        if ex.label not in labels:
            labels.append(ex.label)
    return labels


def _run(
    fn: Callable[[str], Any],
    *,
    mode: str,
    labels: Sequence[str],
    examples: Sequence[Example],
    queries: Sequence[Query],
    orders: list[tuple[int, ...]],
    render_kwargs: dict[str, Any],
) -> tuple[dict[int, dict[int, Optional[str]]], dict[tuple[int, int], str]]:
    predictions: dict[int, dict[int, Optional[str]]] = {}
    prompts_seen: dict[tuple[int, int], str] = {}
    for qi, query in enumerate(queries):
        per_order: dict[int, Optional[str]] = {}
        for oi, order in enumerate(orders):
            prompt = render_prompt(examples, order, query, **render_kwargs)
            prompts_seen[(qi, oi)] = prompt
            raw = fn(prompt)
            if mode == "score":
                if not isinstance(raw, dict):
                    raise ValueError(
                        "score mode callable must return a dict of label -> score, "
                        f"got {type(raw).__name__}"
                    )
                check_finite_scores(raw, labels)
                per_order[oi] = max(labels, key=lambda lab: raw[lab])
            else:
                if not isinstance(raw, str):
                    raise ValueError(
                        "generate mode callable must return a string, "
                        f"got {type(raw).__name__}"
                    )
                per_order[oi] = parse_label(raw, labels)
        predictions[qi] = per_order
    return predictions, prompts_seen


def probe(
    fn: Callable[[str], Any],
    examples: Sequence[Example],
    queries: Sequence[Query],
    *,
    mode: str = "score",
    max_orders: Optional[int] = None,
    seed: int = 0,
    labels: Optional[Sequence[str]] = None,
    instruction: str = DEFAULT_INSTRUCTION,
    example_template: str = DEFAULT_EXAMPLE_TEMPLATE,
    query_template: str = DEFAULT_QUERY_TEMPLATE,
    separator: str = DEFAULT_SEPARATOR,
) -> ShotReport:
    """Probe one example set's order stability over a query set.

    The label set defaults to the examples' labels in first-appearance
    order. Pass ``labels`` to fix a wider label set — needed when a
    reduced example set (see :func:`leave_one_out`) no longer contains
    every label: the model's scores are still validated against the full
    set, and gold answers outside the examples' own labels stay legal.
    With gold answers on the queries, the report carries the accuracy
    block; without them it carries the stability block only.
    """
    if mode not in ("score", "generate"):
        raise ValueError(f"mode must be 'score' or 'generate', got {mode!r}")
    examples = list(examples)
    queries = list(queries)
    if not queries:
        raise ValueError("at least one query is required")
    example_labels = _label_set(examples)
    if labels is None:
        labels = example_labels
    else:
        labels = list(labels)
        missing = [lab for lab in example_labels if lab not in labels]
        if missing:
            raise ValueError(
                f"explicit labels {labels} do not cover example labels {missing}"
            )
    if not labels:
        raise ValueError(
            "examples must be non-empty so the label set is defined; "
            "for a zero-shot run use probe_zero_shot with explicit labels"
        )
    for q in queries:
        if q.answer is not None and q.answer not in labels:
            raise ValueError(
                f"query answer {q.answer!r} is not one of the example labels {labels}"
            )
    orders = make_orders(len(examples), max_orders=max_orders, seed=seed)
    render_kwargs = dict(
        instruction=instruction,
        example_template=example_template,
        query_template=query_template,
        separator=separator,
    )
    predictions, _ = _run(
        fn,
        mode=mode,
        labels=labels,
        examples=examples,
        queries=queries,
        orders=orders,
        render_kwargs=render_kwargs,
    )
    config = {
        "mode": mode,
        "seed": seed,
        "max_orders": max_orders,
        "labels": list(labels),
        "n_examples": len(examples),
        "orders": [list(o) for o in orders],
        "exhaustive": len(orders) == _factorial(len(examples)),
        **describe_templates(
            instruction=instruction,
            example_template=example_template,
            query_template=query_template,
            separator=separator,
        ),
    }
    return build_report(
        labels,
        orders,
        predictions,
        answers=[q.answer for q in queries],
        config=config,
    )


def _factorial(k: int) -> int:
    import math

    return math.factorial(k)


def probe_zero_shot(
    fn: Callable[[str], Any],
    labels: Sequence[str],
    queries: Sequence[Query],
    *,
    mode: str = "score",
    instruction: str = DEFAULT_INSTRUCTION,
    query_template: str = DEFAULT_QUERY_TEMPLATE,
    separator: str = DEFAULT_SEPARATOR,
) -> ShotReport:
    """Probe with no examples at all (one order: the empty permutation).

    Exists so a few-shot report can be compared against the same queries
    with the example set removed entirely, through the same machinery.
    """
    if mode not in ("score", "generate"):
        raise ValueError(f"mode must be 'score' or 'generate', got {mode!r}")
    labels = list(labels)
    if not labels:
        raise ValueError("labels must be non-empty")
    queries = list(queries)
    if not queries:
        raise ValueError("at least one query is required")
    for q in queries:
        if q.answer is not None and q.answer not in labels:
            raise ValueError(f"query answer {q.answer!r} is not in labels {labels}")
    orders = [()]
    render_kwargs = dict(
        instruction=instruction,
        example_template=DEFAULT_EXAMPLE_TEMPLATE,
        query_template=query_template,
        separator=separator,
    )
    predictions, _ = _run(
        fn,
        mode=mode,
        labels=labels,
        examples=[],
        queries=queries,
        orders=orders,  # type: ignore[arg-type]
        render_kwargs=render_kwargs,
    )
    config = {
        "mode": mode,
        "seed": None,
        "max_orders": 1,
        "n_examples": 0,
        "orders": [[]],
        "exhaustive": True,
        **describe_templates(
            instruction=instruction,
            example_template=DEFAULT_EXAMPLE_TEMPLATE,
            query_template=query_template,
            separator=separator,
        ),
    }
    return build_report(
        labels,
        orders,
        predictions,
        answers=[q.answer for q in queries],
        config=config,
    )


def leave_one_out(
    fn: Callable[[str], Any],
    examples: Sequence[Example],
    queries: Sequence[Query],
    *,
    mode: str = "score",
    max_orders: Optional[int] = None,
    seed: int = 0,
    instruction: str = DEFAULT_INSTRUCTION,
    example_template: str = DEFAULT_EXAMPLE_TEMPLATE,
    query_template: str = DEFAULT_QUERY_TEMPLATE,
    separator: str = DEFAULT_SEPARATOR,
) -> dict[str, Any]:
    """Measure each example's influence by removing it and re-probing.

    Returns a dict with the full-set report and, per example index, the
    reduced-set report plus the deltas of mean accuracy (when gold labels
    exist) and stability against the full set. A positive accuracy delta
    for a removal means the example was *hurting*: the set scores better
    without it. Costs one extra probe per example.
    """
    examples = list(examples)
    if len(examples) < 2:
        raise ValueError("leave-one-out needs at least two examples")
    full_labels = _label_set(examples)
    kwargs = dict(
        mode=mode,
        max_orders=max_orders,
        seed=seed,
        labels=full_labels,
        instruction=instruction,
        example_template=example_template,
        query_template=query_template,
        separator=separator,
    )
    full = probe(fn, examples, queries, **kwargs)
    entries: list[dict[str, Any]] = []
    for i in range(len(examples)):
        reduced = examples[:i] + examples[i + 1 :]
        rep = probe(fn, reduced, queries, **kwargs)
        entry: dict[str, Any] = {
            "removed_index": i,
            "removed_example": {"text": examples[i].text, "label": examples[i].label},
            "report": rep.to_dict(),
            "delta_stability": rep.stability - full.stability,
            "delta_mean_accuracy": (
                rep.mean_accuracy - full.mean_accuracy
                if rep.mean_accuracy is not None and full.mean_accuracy is not None
                else None
            ),
        }
        entries.append(entry)
    return {"full": full.to_dict(), "leave_one_out": entries}
