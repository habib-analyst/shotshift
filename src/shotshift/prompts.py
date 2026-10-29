"""Order sets and prompt rendering.

An *order* is a permutation of example indices. Order 0 is always the
identity order — the example set exactly as the caller supplied it — and
every metric that speaks of a "canonical" prediction means the prediction
under order 0.

When the full permutation set fits the budget (``k! <= max_orders``) all
orders are used. Otherwise a seeded sample of distinct non-identity
permutations is drawn. Either way the exact order set is recorded in the
report config, so a sampled run can never masquerade as an exhaustive one.
"""

from __future__ import annotations

import itertools
import math
import random
from typing import Optional, Sequence

from .core import Example, Query

DEFAULT_INSTRUCTION = ""
DEFAULT_EXAMPLE_TEMPLATE = "{text}\nLabel: {label}"
DEFAULT_QUERY_TEMPLATE = "{text}\nLabel:"
DEFAULT_SEPARATOR = "\n\n"


def make_orders(
    k: int, max_orders: Optional[int] = None, seed: int = 0
) -> list[tuple[int, ...]]:
    """Return the order set for ``k`` examples.

    Identity first, then either all remaining permutations (when
    ``k! <= max_orders``) or a seeded sample of ``max_orders - 1`` distinct
    non-identity permutations. ``max_orders=None`` means exhaustive, which
    is refused for k > 8 to avoid an accidental 9! blow-up; pass an
    explicit budget instead.
    """
    if k < 0:
        raise ValueError("k must be >= 0")
    if k <= 1:
        return [tuple(range(k))]
    total = math.factorial(k)
    if max_orders is None:
        if k > 8:
            raise ValueError(
                f"exhaustive probing of {k} examples is {total} orders; "
                "pass max_orders to sample instead"
            )
        budget = total
    else:
        if max_orders < 1:
            raise ValueError("max_orders must be >= 1")
        budget = min(max_orders, total)
    identity = tuple(range(k))
    if budget >= total:
        rest = [p for p in itertools.permutations(range(k)) if p != identity]
        return [identity, *rest]
    rng = random.Random(seed)
    seen = {identity}
    sampled: list[tuple[int, ...]] = [identity]
    while len(sampled) < budget:
        perm = tuple(rng.sample(range(k), k))
        if perm not in seen:
            seen.add(perm)
            sampled.append(perm)
    return sampled


def _render(template: str, *, text: str, label: Optional[str] = None) -> str:
    out = template.replace("{text}", text)
    if label is not None:
        out = out.replace("{label}", label)
    return out


def render_prompt(
    examples: Sequence[Example],
    order: Sequence[int],
    query: Query,
    *,
    instruction: str = DEFAULT_INSTRUCTION,
    example_template: str = DEFAULT_EXAMPLE_TEMPLATE,
    query_template: str = DEFAULT_QUERY_TEMPLATE,
    separator: str = DEFAULT_SEPARATOR,
) -> str:
    """Render one (order, query) prompt.

    The prompt is: the instruction (if any), the examples in the given
    order, and the query rendered with the query template (which normally
    ends at the point where the label would be produced).
    """
    if sorted(order) != list(range(len(examples))):
        raise ValueError("order must be a permutation of the example indices")
    parts: list[str] = []
    if instruction:
        parts.append(instruction)
    for idx in order:
        ex = examples[idx]
        parts.append(_render(example_template, text=ex.text, label=ex.label))
    parts.append(_render(query_template, text=query.text))
    return separator.join(parts)


def describe_templates(
    *,
    instruction: str,
    example_template: str,
    query_template: str,
    separator: str,
) -> dict[str, str]:
    """The rendering configuration, recorded verbatim in report configs."""
    return {
        "instruction": instruction,
        "example_template": example_template,
        "query_template": query_template,
        "separator": separator,
    }
