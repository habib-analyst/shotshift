from shotshift import parse_label

LABELS = ["positive", "negative"]


def test_marker_form():
    assert parse_label("Label: positive", LABELS) == "positive"
    assert parse_label("Answer: negative.", LABELS) == "negative"
    assert parse_label("The sentiment is clear.\nSentiment: positive", LABELS) == "positive"


def test_leading_label():
    assert parse_label("positive", LABELS) == "positive"
    assert parse_label("Negative — the review is scathing.", LABELS) == "negative"
    assert parse_label("positive.", LABELS) == "positive"


def test_bare_label_short_response():
    assert parse_label("I would say positive", LABELS) == "positive"


def test_refusals():
    # both labels at the same precedence level (bare) -> never guess
    assert parse_label("I think positive or negative both fit", LABELS) is None
    # label as substring of a longer word must not match
    assert parse_label("Label: positively glowing", LABELS) is None
    # long response without marker/leading label is not mined for words
    long_text = "This review discusses many things at length and somewhere " \
                "in the middle a positive word appears but it is not a verdict"
    assert parse_label(long_text, LABELS) is None
    assert parse_label("", LABELS) is None
    assert parse_label("no verdict here", LABELS) is None


def test_marker_with_two_labels_is_refused():
    assert parse_label("Label: positive / negative", LABELS) is None
