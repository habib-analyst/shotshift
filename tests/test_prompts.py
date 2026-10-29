import itertools

import pytest

from shotshift import Example, Query, make_orders, render_prompt
from shotshift.prompts import DEFAULT_SEPARATOR


def test_orders_identity_first_and_exhaustive_for_small_k():
    orders = make_orders(3)
    assert orders[0] == (0, 1, 2)
    assert len(orders) == 6
    assert set(orders) == set(itertools.permutations(range(3)))


def test_orders_sampled_are_distinct_seeded_and_identity_first():
    a = make_orders(8, max_orders=16, seed=7)
    b = make_orders(8, max_orders=16, seed=7)
    assert a == b  # seeded determinism
    assert a[0] == tuple(range(8))
    assert len(a) == 16
    assert len(set(a)) == 16
    assert all(sorted(o) == list(range(8)) for o in a)


def test_orders_budget_above_factorial_is_exhaustive():
    assert len(make_orders(4, max_orders=1000)) == 24


def test_orders_edge_cases():
    assert make_orders(0) == [()]
    assert make_orders(1) == [(0,)]
    with pytest.raises(ValueError):
        make_orders(9)  # exhaustive refused without a budget
    with pytest.raises(ValueError):
        make_orders(4, max_orders=0)


def test_render_prompt_places_examples_in_given_order():
    examples = [Example("alpha", "positive"), Example("beta", "negative")]
    query = Query("gamma")
    p_identity = render_prompt(examples, (0, 1), query)
    p_swapped = render_prompt(examples, (1, 0), query)
    assert p_identity.index("alpha") < p_identity.index("beta")
    assert p_swapped.index("beta") < p_swapped.index("alpha")
    assert p_identity.endswith("gamma\nLabel:")
    assert "alpha\nLabel: positive" in p_identity


def test_render_prompt_with_instruction_and_validation():
    examples = [Example("alpha", "positive")]
    out = render_prompt(
        examples, (0,), Query("q"), instruction="Classify the sentiment."
    )
    assert out.startswith("Classify the sentiment." + DEFAULT_SEPARATOR)
    with pytest.raises(ValueError):
        render_prompt(examples, (0, 0), Query("q"))
    with pytest.raises(ValueError):
        render_prompt(examples, (1,), Query("q"))
