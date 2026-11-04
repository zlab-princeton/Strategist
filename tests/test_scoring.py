"""The generic grader: the gold answer's form decides the rule, and a match is
exact after extraction, never by containment.

    python -m pytest tests/test_scoring.py -q
"""
from __future__ import annotations

from strategist.scoring import grade, mean_score

MCQ = "Which is a prime? (A) 4 (B) 7 (C) 9 (D) 15. Answer with just the letter."


def test_letter_answers():
    assert grade("t", MCQ, "B", "The prime is 7, so the answer is B.")["passed"]
    assert grade("t", MCQ, "B", "**Answer:** B")["passed"]
    assert not grade("t", MCQ, "B", "I think (C) 9.")["passed"]
    assert not grade("t", MCQ, "B", "B or C")["passed"]


def test_yes_no_answers():
    q = "Is 7 prime? Answer with just 'yes' or 'no'."
    assert grade("t", q, "yes", "Yes.")["passed"]
    assert grade("t", q, "no", "No, it is not.")["passed"]
    assert not grade("t", q, "yes", "no")["passed"]


def test_numeric_answers_compare_as_numbers():
    q = "What is 250 * 4?"
    assert grade("t", q, "1000", "250 * 4 = 1,000")["passed"]
    assert grade("t", q, "1000", "#### 1000.0")["passed"]
    assert not grade("t", q, "1000", "The answer is 100")["passed"]


def test_boxed_latex_answers():
    q = "Solve for x."
    assert grade("t", q, "\\frac{1}{2}", "So $x = \\boxed{\\frac{1}{2}}$.")["passed"]
    assert grade("t", q, "\\frac{1}{2}", "x = \\boxed{1/2}")["passed"]
    assert not grade("t", q, "\\frac{1}{2}", "x = \\boxed{2}")["passed"]


def test_text_answers_use_the_references():
    q = "Which city hosts the Louvre? Answer concisely."
    assert grade("t", q, "Paris", "Answer: Paris")["passed"]
    assert grade("t", q, "Paris", "paris, france", refs=["Paris, France"])["passed"]
    assert not grade("t", q, "Paris", "Lyon", refs=["Paris, France"])["passed"]
    assert not grade("t", q, "Paris", "Paris is lovely, but the answer is Lyon")["passed"]


def test_empty_and_missing_values_never_pass():
    assert grade("t", MCQ, "B", "")["note"] == "empty_output"
    assert grade("t", MCQ, "", "B")["note"] == "no_expected"
    assert not grade("t", MCQ, None, "B")["passed"]


def test_thinking_prefix_is_ignored():
    out = "<think>maybe C... no, 7 is prime</think>\nThe answer is B."
    assert grade("t", MCQ, "B", out)["passed"]


def test_mean_score_reads_scores_then_passed_flags():
    records = [{"passed": True}, {"passed": False}, {"score": 0.5}, {"score": 1.0, "passed": False}]
    assert mean_score(records) == (1 + 0 + 0.5 + 1.0) / 4
    assert mean_score([]) == 0.0
