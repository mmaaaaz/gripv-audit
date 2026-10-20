"""Unit tests for option matching — no dataset needed."""

from gripv.engine import Answer, letter, match_option, norm


def test_norm_strips_articles_and_punctuation():
    assert norm("The dotted jug!") == "dotted jug"


def test_exact_text_match():
    assert match_option("the dotted jug", ["crosshatch jug", "dotted jug"]) == 1


def test_numeric_match_with_units():
    assert match_option("5 s", ["4 s", "5 s", "6 s"]) == 1


def test_numeric_tolerance_inside_window():
    assert match_option("1.2 s", ["1.0 s", "1.5 s"], tol=0.25) == 0


def test_numeric_tolerance_outside_window():
    assert match_option("1.2 s", ["1.6 s", "2.0 s"], tol=0.1) is None


def test_missing_answer_returns_none():
    assert match_option(None, ["a", "b"]) is None


def test_letter_boundaries():
    assert letter(0) == "A"
    assert letter(5) == "F"
    assert letter(None) is None


def test_answer_defaults_are_explicit():
    a = Answer("3", confidence="derived", note="computed from two events")
    assert a.strict == "3"
    assert a.loose is None
    assert a.confidence == "derived"
