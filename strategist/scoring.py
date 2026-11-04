"""Grading of a strategy's output against a task's gold answer.

The gold answer's form decides the rule: yes/no, a choice letter, a choice index,
a top-5 ranking, a number, or text, each compared after answer extraction and
normalisation, never by containment. A row's `answers` list accepts any of its
references. Code tasks are executed by the search loop instead
(common._evaluate_humaneval_completion).
"""
from __future__ import annotations

import re

from strategist.workflows._qa_common import (
    NUMBER_RE,
    extract_answer,
    extract_answer_nq_open,
    _strip_string,
    extract_letter,
    extract_yesno,
    last_boxed_only_string,
    remove_boxed,
    task_kind,
)


def _norm_text(value: str) -> str:
    """Lowercase, strip punctuation at edges, collapse whitespace."""
    value = (value or "").casefold().strip()
    value = re.sub(r"^[\"'`\s]+|[\"'`\s.,!?;:]+$", "", value)
    value = re.sub(r"\s+", " ", value)
    return value


_MD_ANSWER_LABEL = re.compile(r"^[ \t>#*+\-]*((?:final[ \t]+)?answer)[ \t*]*:[ \t*]*(.*)$", re.IGNORECASE | re.MULTILINE)


def _plain_answer_labels(output: str) -> str:
    """Markdown around an answer label is dropped so it reads like the plain
    "Answer: X" the extractor looks for (until 2026-10-13 a bold label was not
    recognised and the label words were graded as part of the answer)."""
    return _MD_ANSWER_LABEL.sub(
        lambda m: f"{m.group(1)}: {m.group(2).replace('**', '').strip().strip('*').strip()}", output)


def _final_text(output: str) -> str:
    """The final answer of a free-text response; a whole-answer \\boxed{...} is unwrapped."""
    final = extract_answer_nq_open(_plain_answer_labels(output))
    if "\\boxed" in final:
        try:
            inner = remove_boxed(last_boxed_only_string(final))
            if inner:
                final = re.sub(r"^\\text\{(.*)\}$", r"\1", inner.strip())
        except Exception:
            pass
    return final.strip().strip("$").strip()


def _final_number(output: str) -> float | None:
    body = (output or "").replace(",", "")
    tagged = re.findall(r"####\s*(-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)", body)
    candidates = tagged or NUMBER_RE.findall(body)
    if not candidates:
        return None
    try:
        return float(candidates[-1])
    except ValueError:
        return None


def _grade_letter(question: str, expected: str, output: str) -> tuple[float, str]:
    m = re.search(r"[A-Za-z]", expected)
    if not m:
        return 0.0, "expected_no_letter"
    extracted = extract_letter(question, output)
    return (1.0, "") if extracted == m.group(0).upper() else (0.0, f"extracted={str(extracted)[:40]}")


def extract_choice_index(question: str, output: str) -> str | None:
    """Option index for benchmarks whose options are numbered (HellaSwag 0-3,
    WinoGrande 1-2). Only digits that are offered as options count."""
    valid = set(re.findall(r"^\(?([0-9])\)?[:.)]? ", question or "", flags=re.M)) or set("0123")
    body = (output or "").strip()
    m = re.fullmatch(r"\(?\s*([0-9])\s*\)?\.?", body)
    if m:
        return m.group(1) if m.group(1) in valid else None
    marked = re.findall(
        r"(?i:(?:final\s+)?answer(?:\s+is)?|continuation\s+is)\s*[:=]?\s*\**\s*\(?([0-9])\)?(?!\d|\.\d)", body
    )
    marked = [d for d in marked if d in valid]
    if marked:
        return marked[-1]
    alone = [d for d in re.findall(r"(?<![\d.])([0-9])(?!\d|\.\d)", body) if d in valid]
    return alone[-1] if alone else None


def _grade_index(question: str, expected: str, output: str) -> tuple[float, str]:
    got = extract_choice_index(question, output)
    return (1.0, "") if got == expected.strip() else (0.0, f"extracted={got}")


def _grade_yesno(expected: str, output: str) -> tuple[float, str]:
    gold = "yes" if expected.strip().lower() in {"yes", "true"} else "no"
    got = extract_yesno(output)
    return (1.0, f"got={got}") if got == gold else (0.0, f"got={got}")


def _grade_numeric(expected: str, output: str, *, integer_only: bool = False) -> tuple[float, str]:
    exp_raw = expected.strip().replace(",", "")
    parts = [p.strip() for p in exp_raw.split(";") if p.strip()]
    if len(parts) > 1:
        try:
            gold = sorted(float(p) for p in parts)
        except ValueError:
            gold = None
        if gold is not None:
            # Multi-part ASDiv answer ("40 (years old); 10 (years old)"): the
            # response has to end with exactly these numbers, in any order.
            nums = NUMBER_RE.findall((output or "").replace(",", ""))[-len(gold):]
            try:
                got = sorted(float(n) for n in nums)
            except ValueError:
                got = []
            ok = len(got) == len(gold) and all(
                abs(a - b) <= 1e-4 * max(1.0, abs(b)) for a, b in zip(got, gold)
            )
            return (1.0 if ok else 0.0), f"got={got}"
    try:
        exp_num = float(exp_raw)
    except ValueError:
        # A few ASDiv golds are ratios or names: compare the final answer as text.
        got = _final_text(output)
        return (1.0, "") if _norm_text(got) == _norm_text(expected) else (0.0, f"extracted={got[:40]}")
    got_num = _final_number(output)
    if got_num is None:
        return 0.0, "no_number"
    if integer_only or ("." not in exp_raw and "e" not in exp_raw.lower() and exp_num.is_integer()):
        return (1.0 if got_num == exp_num else 0.0), f"got={got_num}"
    ok = abs(got_num - exp_num) <= 1e-4 * max(1.0, abs(exp_num))
    return (1.0 if ok else 0.0), f"got={got_num}"


def _grade_hit5(expected: str, output: str) -> tuple[float, str]:
    """Amazon-Rec hit@5: the gold item id is among the five recommended ids.
    The task asks for exactly five numbers on the last line."""
    lines = [ln for ln in (output or "").splitlines() if ln.strip()]
    last = re.findall(r"\b\d+\b", lines[-1]) if lines else []
    nums = last[:5] if len(last) >= 5 else re.findall(r"\b\d+\b", output or "")[:5]
    return (1.0 if expected.strip() in nums else 0.0), f"top5={','.join(nums)}"


def _grade_generic(question: str, expected: str, output: str) -> tuple[float, str]:
    """Dispatch on the gold answer's form and the task text (exact match, never
    containment)."""
    if expected.lower() in {"yes", "no", "true", "false"}:
        return _grade_yesno(expected, output)
    if re.fullmatch(r"\(?\s*[A-Za-z]\s*\)?\.?", expected):
        return _grade_letter(question, expected, output)
    kind = task_kind(question)
    if kind == "letter":
        return _grade_letter(question, expected, output)
    if kind == "small_choice":
        return _grade_index(question, expected, output)
    if kind == "ranking":
        return _grade_hit5(expected, output)
    if kind == "codegen":
        return (1.0 if _norm_text(expected) == _norm_text(output) else 0.0), "codegen_textual"
    if re.fullmatch(r"-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?", expected.replace(",", "")):
        return _grade_numeric(expected, output)
    got = extract_answer(question, output)
    if _norm_text(got) == _norm_text(expected):
        return 1.0, ""
    try:
        if _strip_string(got) == _strip_string(expected):
            return 1.0, "math_equiv"
    except Exception:
        pass
    return 0.0, f"extracted={got[:60]}"


# ----------------------------------------------------------------- entry points

def grade(benchmark: str, question: str, expected, output, refs=None) -> dict:
    """Grade one output. Returns {"passed": bool, "score": float, "note": str}.
    `expected` is the gold answer (a string or a list of accepted strings);
    `refs` lists further accepted answers."""
    golds = [expected] if isinstance(expected, str) else [str(x) for x in (expected or [])]
    golds += [str(r) for r in (refs or [])]
    golds = [g.strip().replace("\u2212", "-") for g in golds if g and g.strip()]
    if not golds:
        return {"passed": False, "score": 0.0, "note": "no_expected"}
    if output is None or str(output).strip() == "":
        return {"passed": False, "score": 0.0, "note": "empty_output"}
    raw_output = str(output)
    if "</think>" in raw_output and raw_output.rsplit("</think>", 1)[-1].strip():
        raw_output = raw_output.rsplit("</think>", 1)[-1]
    raw_output = raw_output.replace("\u2212", "-")
    question = question or ""
    best, note = -1.0, ""
    for gold in golds:
        score, gold_note = _grade_generic(question, gold, raw_output)
        if score > best:
            best, note = score, gold_note
        if best >= 1.0:
            break
    return {"passed": bool(best >= 1.0), "score": float(max(best, 0.0)), "note": note}


def score_output(benchmark: str, question: str, expected, output) -> tuple[bool, str]:
    """Score a strategy's output against the reference. Returns (passed, note)."""
    g = grade(benchmark, question, expected, output)
    return g["passed"], g["note"]


def sample_score(sample: dict) -> float:
    """Per-sample score of a stored record: `score` when present (a partial score),
    otherwise 1.0 / 0.0 from `passed`."""
    if isinstance(sample.get("score"), (int, float)):
        return float(sample["score"])
    return 1.0 if sample.get("passed") else 0.0


def mean_score(samples: list[dict]) -> float:
    """Accuracy of a list of sample records (mean per-sample score)."""
    return sum(sample_score(s) for s in samples) / len(samples) if samples else 0.0
