"""Shared QA helpers used by the paper-faithful baseline workflows.

This module is private to the workflow layer (note the underscore prefix) — it
holds task-type detection, prompt fragments, answer extraction, and multi-sample
aggregation utilities so that each new workflow can stay focused on its
technique-level logic.
"""
from __future__ import annotations

import asyncio
import re
import string
from collections import Counter

from strategist.basic import generate


LETTER_RE = re.compile(r"\b([A-Z])\b")
SMALL_CHOICE_RE = re.compile(r"\b([0-3]|1|2)\b")
NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?")
ITEM_NUMBER_RE = re.compile(r"\b\d+\b")
# Only the rest of the fence line is skipped: the indentation of the first code
# line belongs to the code (a body-only HumanEval completion starts indented).
CODE_FENCE_RE = re.compile(r"```(?:python|py)?[ \t]*\n?(.*?)```", re.DOTALL | re.IGNORECASE)


def _looks_like_letter_task(question: str) -> bool:
    lowered = question.lower()
    return (
        "answer with just the letter" in lowered
        or "respond with just the letter" in lowered
        or ("just the letter" in lowered and "letter" in lowered)
    )


def _looks_like_small_choice_task(question: str) -> bool:
    lowered = question.lower()
    return (
        "answer with just the number (0-3)" in lowered
        or "fill in the blank with option 1 or 2" in lowered
        or "answer with just 1 or 2" in lowered
        or "answer with just 1" in lowered
        or "answer with just 2" in lowered
    )


def _looks_like_ranking_task(question: str) -> bool:
    lowered = question.lower()
    return "top 5 items" in lowered or "exactly 5 numbers separated by commas" in lowered


def _looks_like_codegen_task(question: str) -> bool:
    # HumanEval tasks start with "Complete this python function:"; MBPP and
    # MBPP+ tasks start with "Write a Python function". Without the second
    # form an MBPP response was treated as a numeric answer and reduced to
    # the last number in the code.
    q = question.lower()
    return q.startswith("complete this python function:") or q.startswith(
        "write a python function"
    )


def task_kind(question: str) -> str:
    if _looks_like_codegen_task(question):
        return "codegen"
    if _looks_like_ranking_task(question):
        return "ranking"
    if _looks_like_letter_task(question):
        return "letter"
    if _looks_like_small_choice_task(question):
        return "small_choice"
    return "numeric"


def answer_kind(question: str) -> str:
    """Finer answer type behind the shared extractor (paper A.2: option letter,
    integer, LaTeX numeric, span, yes/no).

    `task_kind` keeps its coarse legacy values because library strategies branch
    on them; this refines the old catch-all "numeric" by the fixed instruction
    line each benchmark builder appends to the task:
      yesno   "Answer with just 'yes' or 'no'."      (StrategyQA, BoolQ)
      text    "Answer concisely."                     (SQuAD, HotpotQA, NQ, ...)
      final   "Give just the final answer."           (BBH)
      literal "Respond with just the output value"    (CRUXEval)
      math    a bare question                         (GSM8K, MATH, MinervaMath, ...)
    """
    kind = task_kind(question)
    if kind != "numeric":
        return kind
    tail = question.strip().lower()[-240:]
    if "answer with just 'yes' or 'no'" in tail or 'answer with just "yes" or "no"' in tail:
        return "yesno"
    if tail.endswith("answer concisely."):
        return "text"
    if tail.endswith("give just the final answer."):
        return "final"
    if "respond with just the output value" in tail:
        return "literal"
    return "math"


_YESNO_TOKEN_RE = re.compile(r"\b(yes|no|true|false)\b", re.IGNORECASE)
_YESNO_MARKER_RE = re.compile(
    r"(?:final\s+)?answer(?:\s+here)?(?:\s+is)?\s*[:=]?[\s\*\"'“”‘’(\[]*(yes|no|true|false)\b", re.IGNORECASE
)
_YESNO_LINE_RE = re.compile(r"[\s\*\"'“”‘’(\[]*(yes|no|true|false)[\s\*\"'“”‘’)\].!]*", re.IGNORECASE)


def extract_yesno(text: str) -> str | None:
    """'yes' / 'no' from a response. Order: a last line that is only the
    verdict, an explicit answer marker, a response that opens with the verdict,
    and finally the last yes/no token. The last line goes first: reasoning such
    as "the candidate answer 'no' is eliminated" must not outrank a closing "yes"."""
    body = (text or "").strip()
    if "</think>" in body and body.rsplit("</think>", 1)[-1].strip():
        body = body.rsplit("</think>", 1)[-1].strip()
    if not body:
        return None
    canon = lambda tok: "yes" if tok.lower() in {"yes", "true"} else "no"
    last_line = body.splitlines()[-1]
    m = _YESNO_LINE_RE.fullmatch(last_line)
    if m:
        return canon(m.group(1))
    marked = _YESNO_MARKER_RE.findall(body)
    if marked:
        return canon(marked[-1])
    lead = re.match(r"[\s\*\"'“‘(\[]*(yes|no|true|false)\b", body, re.IGNORECASE)
    if lead:
        return canon(lead.group(1))
    tokens = _YESNO_TOKEN_RE.findall(body)
    return canon(tokens[-1]) if tokens else None


# A comma only counts as a thousands separator ("1,234"); "1,2" is a list of two answers.
_PLAIN_NUMBER_RE = re.compile(
    r"^\\?\$?\s*(-?(?:(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?|\.\d+)(?:[eE][-+]?\d+)?)\s*(?:\\?%)?$"
)
_NUMBER_LIST_RE = re.compile(r"\s*-?\d+(?:\.\d+)?(?:\s*,\s*-?\d+(?:\.\d+)?)+\s*")
_RATIO_RE = re.compile(r"\s*\d+(?:\.\d+)?(?:\s*:\s*\d+(?:\.\d+)?)+\s*")  # "3:1", "5:30"
_EXPRESSION_HINT_RE = re.compile(r"[\\^_{}=()/+*\[\]]|\d[a-zA-Z]|[a-zA-Z]\d|\\pi|\bpi\b|sqrt")
# "b - 1", "1 - x": single-letter variables joined by an operator, no words.
_VARIABLE_EXPR_RE = re.compile(r"[\d\s.+\-*/^()a-zA-Z]{1,40}")
_MATH_MARKER_RE = re.compile(
    r"(?:final\s+answer|answer)\s*(?:is|=|:)[\s:=\*]*(.+?)\s*(?:\n|$)", re.IGNORECASE
)


def _strip_math_wrappers(text: str) -> str:
    s = (text or "").strip()
    s = re.sub(r"\s*I hope it is correct\.?\s*$", "", s, flags=re.IGNORECASE).strip()
    s = s.strip("*").strip()
    if s.endswith(".") and not re.search(r"\d\.$", s[-3:]):
        s = s[:-1].strip()
    for left, right in (("$$", "$$"), ("\\[", "\\]"), ("\\(", "\\)")):
        if s.startswith(left) and s.endswith(right) and len(s) > len(left) + len(right):
            s = s[len(left):-len(right)].strip()
    if s.startswith("$") and s.endswith("$") and s.count("$") == 2 and len(s) > 2:
        s = s[1:-1].strip()
    return s


def _as_plain_number(text: str) -> str | None:
    """'1,234.50' / '$18' / '4.5e33' -> canonical number string; None otherwise."""
    cleaned = (text or "").replace("\\!", "").replace("\\,", "").strip()
    m = _PLAIN_NUMBER_RE.match(cleaned)
    return m.group(1).replace(",", "") if m else None


def _answer_from_candidate(candidate: str) -> str:
    cand = _strip_math_wrappers(candidate)
    plain = _as_plain_number(cand)
    if plain is not None:
        return plain
    if _EXPRESSION_HINT_RE.search(cand):
        return cand  # LaTeX / symbolic / scientific-notation answer: keep it whole
    if _NUMBER_LIST_RE.fullmatch(cand) or _RATIO_RE.fullmatch(cand):
        return cand  # several answers ("1,-2", "3, 5, 7"): keep the list
    if (_VARIABLE_EXPR_RE.fullmatch(cand) and not re.search(r"[A-Za-z]{2,}", cand)
            and re.search(r"[A-Za-z]", cand) and re.search(r"[-+*/^]", cand)):
        return cand  # symbolic expression in one-letter variables
    numbers = NUMBER_RE.findall(cand.replace(",", ""))
    return numbers[-1] if numbers else cand  # "18 dollars" -> "18"


def _math_candidate(body: str) -> str | None:
    """Final-answer text of a math response before any reduction: the last
    \\boxed{...}, an explicit answer marker, or a short one-line response."""
    if "\\boxed" in body or "\\fbox" in body:
        try:
            boxed = remove_boxed(last_boxed_only_string(body))
        except Exception:
            boxed = None
        if boxed:
            return boxed
    marked = _MATH_MARKER_RE.findall(body)
    if marked and marked[-1].strip():
        return marked[-1]
    if "\n" not in body and len(body) <= 200:
        return body
    # A longer single line that is one LaTeX expression (no sentence break).
    if "\n" not in body and len(body) <= 400 and "\\" in body and ". " not in body:
        return body
    return None


def math_answer_raw(text: str) -> str:
    """The same final-answer text as extract_math_answer, without reducing it to
    a number ("7.6 Z" stays "7.6 Z"). Empty when the response has no boxed
    answer, marker or one-line form."""
    body = (text or "").strip().replace("\u2212", "-")
    cand = _math_candidate(body) if body else None
    return _strip_math_wrappers(cand) if cand else ""


def extract_math_answer(text: str) -> str:
    """Final answer of a math / numeric response without destroying symbolic
    answers. Order: GSM8K '####' tag, last \\boxed{...}, explicit answer
    marker, a short one-line response, and only then the last number."""
    body = (text or "").strip().replace("\u2212", "-")  # Unicode minus
    if not body:
        return body
    tagged = re.findall(r"####\s*([-\d,.\s]+)", body)
    if tagged and tagged[-1].strip():
        return tagged[-1].strip().replace(",", "")
    cand = _math_candidate(body)
    if cand is not None:
        return _answer_from_candidate(cand)
    numbers = NUMBER_RE.findall(body.replace(",", ""))
    return numbers[-1] if numbers else body


def output_instruction(question: str) -> str:
    kind = task_kind(question)
    if kind == "codegen":
        return "Return only Python code."
    if kind == "ranking":
        return "Output exactly 5 item numbers separated by commas as the very last line, and nothing after."
    if kind == "letter":
        return (
            "The very last token of your response MUST be the single correct "
            "option letter — nothing after it. If you write "
            "any reasoning, the final letter must be the answer."
        )
    if kind == "small_choice":
        return (
            "The very last token of your response MUST be the required option "
            "number — nothing after it."
        )
    return (
        "The very last line of your response MUST be the final answer only — "
        "nothing after it. If you write reasoning, put the final answer on "
        "the last line by itself."
    )


def base_system(question: str) -> str:
    kind = task_kind(question)
    if kind == "codegen":
        return "Complete the Python function. Keep the signature intact and return only code."
    if kind == "ranking":
        return "You are a recommendation assistant. Return only the requested item ids."
    return "Answer the question carefully and follow the requested output format exactly."


def extract_letter(question: str, text: str) -> str | None:
    """Extract an MCQ option letter from `text`, independent of how the
    question is phrased (BBH stores letter golds without a letter
    instruction). Returns None when no plausible letter is found.

    Order: bare single letter; explicit answer marker ("Answer: B",
    "the answer is (b)" — uppercase trusted, lowercase only when terminal so
    the article in "the answer is a tricky one" is never option A); last
    standalone uppercase letter restricted to the question's offered options.
    Never uppercases the whole body first — that manufactures fake "A"/"I"
    matches from articles and pronouns.
    """
    body = (text or "").strip()
    if len(body) == 1 and body.upper() in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
        return body.upper()
    # Bare decorated letter as the whole output: "(b)", "C.", "(D)".
    m = re.fullmatch(r"\(\s*([A-Za-z])\s*\)\.?|([A-Za-z])\.", body)
    if m:
        return (m.group(1) or m.group(2)).upper()
    prefix = r"(?i:(?:final\s+)?answer(?:\s+is)?)\s*[:=]?\s*"
    for pat in (
        prefix + r"\(([A-Za-z])\)",
        prefix + r"([A-Z])\b",
        prefix + r"([a-z])(?=\s*(?:[).,!?:;]|\n|$))",
    ):
        marked = re.findall(pat, body)
        if marked:
            return marked[-1].upper()
    options = set(re.findall(r"\(([A-Z])\)", question)) or set("ABCDEFGH")
    matches = [m for m in LETTER_RE.finditer(body) if m.group(1) in options]
    return matches[-1].group(1) if matches else None


def extract_answer(question: str, text: str) -> str:
    kind = task_kind(question)
    body = (text or "").strip()

    if kind == "codegen":
        # The whole code block is returned for every code task. The evaluator
        # places a full function after the prompt and attaches a bare body to
        # the prompt's signature, so nothing is cut here. (An earlier version
        # removed the first `def` line, which broke completions whose first
        # def is a helper.)
        match = CODE_FENCE_RE.search(body)
        return match.group(1).strip("\n").rstrip() if match else body

    if kind == "ranking":
        numbers = ITEM_NUMBER_RE.findall(body)
        return ", ".join(numbers[:5]) if numbers else body

    if kind == "letter":
        return extract_letter(question, body) or body

    if kind == "small_choice":
        matches = list(SMALL_CHOICE_RE.finditer(body))
        return matches[-1].group(1) if matches else body

    # The old code took "the last number in the response" for every remaining
    # task. That rewrote text answers ("February 2005" -> "2005"), scientific
    # notation ("4.5e33" -> "33") and LaTeX answers into bare numbers before
    # grading. Dispatch on the answer type instead.
    fine = answer_kind(question)
    if fine == "yesno":
        return extract_yesno(body) or body
    if fine in {"text", "final", "literal"}:
        return extract_answer_nq_open(body)
    return extract_math_answer(body)


# ── NQ-open specific extraction ──────────────────────────────────────────────
# The default extract_answer uses NUMBER_RE which grabs the last digit from
# "1. Answer text" format (returning "1" instead of "Answer text").
# For factual open QA (NQ-style), the answer is text, not a number.


def extract_answer_nq_open(text: str) -> str:
    """Extract factual answer for open QA (NQ-style). Tries explicit markers first, then freeform text."""
    body = (text or "").strip()
    if not body:
        return ""

    # Priority 1: "Answer: <text>" — strip everything before "answer:"
    answer_match = re.search(
        r"(?:^|\n)Answer:\s*(.+?)(?:\n|$)", body, re.IGNORECASE
    )
    if answer_match:
        val = answer_match.group(1).strip()
        # Remove list-number prefixes (1. or 1) followed by space) but NOT decimals
        val = re.sub(r"^\d{1,2}[\.\)]\s+(?=[^\d])", "", val)
        val = re.sub(r"^Final\s+Answer:\s*", "", val, flags=re.IGNORECASE)
        if val:
            return val

    # Priority 2: "Final Answer: <text>" — strip "Final Answer:" prefix
    final_match = re.search(
        r"(?:^|\n)(?:Final\s+)?Answer:\s*(.+?)(?:\n|$)", body, re.IGNORECASE
    )
    if final_match:
        val = final_match.group(1).strip()
        val = re.sub(r"^\d{1,2}[\.\)]\s+(?=[^\d])", "", val)
        if val:
            return val

    # Priority 2b: prose marker "the answer is <text>" (no colon) — common in
    # model output; take the LAST occurrence and strip a trailing period.
    prose_matches = re.findall(
        r"(?:the\s+)?(?:final\s+)?answer\s+is\s*[:=]?\s*(.+?)(?:\n|$)",
        body,
        re.IGNORECASE,
    )
    if prose_matches:
        val = prose_matches[-1].strip().rstrip(".")
        if val:
            return val

    # Priority 3: Strip numbered/bulleted list prefixes ("1. foo" -> "foo"),
    # but never mangle a bare numeric answer: "1.5" must stay "1.5".
    if re.fullmatch(r"-?\d+(?:[\.,]\d+)*", body):
        stripped = body
    else:
        stripped = re.sub(r"^\d{1,2}[\.\)]\s+(?=[^\d])", "", body).strip()

    # Priority 4: strip markdown/formatting noise: ****, ----, ***, etc.
    stripped = re.sub(r"^[*#\-_=~]{2,}\s*", "", stripped)
    # Also strip trailing formatting (e.g. "Answer**" → "Answer")
    stripped = re.sub(r"\s*[*#\-_=~]{2,}$", "", stripped)

    # Priority 5: if the whole thing looks like a short answer (under 30 words,
    # no paragraph breaks), return it directly
    if len(stripped.split()) <= 30 and "\n" not in stripped:
        return stripped

    # Priority 6: grab the last non-empty content line, skipping instruction echoes
    lines = [l.strip() for l in body.splitlines() if l.strip()]
    for line in reversed(lines):
        # Skip instruction-echo lines (e.g. "Answer concisely.", "Reason step by step.")
        if re.match(r"^(Answer\s+(concisely|step\s*by\s*step|the\s+following|with\s+care)|Reasoning|Step\s+\d+:?\s*$)", line, re.IGNORECASE):
            continue
        # For labeled answer lines, extract content after the label
        if re.match(r"^(Final\s+)?Answer:", line, re.IGNORECASE):
            val = re.sub(r"^(?:Final\s+)?Answer:\s*", "", line, flags=re.IGNORECASE)
            val = re.sub(r"^\d{1,2}[\.\)]\s*", "", val)
            if val:
                return val
        # For all other lines, return the last non-empty content line
        return line

    return stripped if stripped else body

    return stripped if stripped else body


def _normalize_for_vote(question: str, answer: str) -> str:
    kind = task_kind(question)
    body = (answer or "").strip()
    if kind == "letter":
        return body.upper()
    if kind == "small_choice":
        return body
    if kind == "ranking":
        return ",".join(re.findall(r"\d+", body)[:5])
    if kind == "codegen":
        return re.sub(r"\s+", " ", body)[:200]
    fine = answer_kind(question)
    if fine == "yesno":
        return extract_yesno(body) or body.lower()
    if fine in {"text", "final", "literal"}:
        return normalize_answer(body) or body.lower()
    try:
        return str(float(body.replace(",", "")))
    except Exception:
        try:
            return _strip_string(body)
        except Exception:
            return body


def majority_vote(question: str, answers: list[str]) -> str:
    if not answers:
        return ""
    normalized: list[tuple[str, str]] = []
    for raw in answers:
        key = _normalize_for_vote(question, raw)
        if key:
            normalized.append((key, raw))
    if not normalized:
        return answers[0]
    counter = Counter(key for key, _ in normalized)
    top_key, _ = counter.most_common(1)[0]
    for key, raw in normalized:
        if key == top_key:
            return raw
    return answers[0]


async def sample_n_answers(
    llm,
    question: str,
    *,
    n: int = 5,
    temperature: float = 0.7,
    max_tokens: int | None = None,
) -> list[str]:
    """Draw N independent CoT samples in parallel and extract an answer from each."""
    kind = task_kind(question)
    prompt = (
        question
        if kind == "codegen"
        else f"{question}\n\nReason step by step if helpful, then give the final answer in the required format."
    )
    system = f"{base_system(question)} {output_instruction(question)}"
    # Paper A.4: every strategy LLM call requests max_tokens=16384.
    tokens = max_tokens or 16384

    async def one() -> str:
        response = await generate(
            llm,
            prompt,
            system=system,
            temperature=temperature,
            max_tokens=tokens,
        )
        return extract_answer(question, response)

    return await asyncio.gather(*(one() for _ in range(max(1, n))))


# ----------------------------------------------------------------- answer normalisation
# Text answers follow the SQuAD evaluation script; LaTeX answers follow the MATH
# dataset's grader (Hendrycks et al., 2021; github.com/hendrycks/math).


def normalize_answer(s: str) -> str:
    """Lower text and remove punctuation, articles and extra whitespace."""
    s = "".join(ch for ch in s.lower() if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def _fix_fracs(string):
    substrs = string.split("\\frac")
    new_str = substrs[0]
    if len(substrs) > 1:
        substrs = substrs[1:]
        for substr in substrs:
            new_str += "\\frac"
            if substr[0] == "{":
                new_str += substr
            else:
                try:
                    assert len(substr) >= 2
                except:
                    return string
                a = substr[0]
                b = substr[1]
                if b != "{":
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}{" + b + "}" + post_substr
                    else:
                        new_str += "{" + a + "}{" + b + "}"
                else:
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}" + b + post_substr
                    else:
                        new_str += "{" + a + "}" + b
    string = new_str
    return string


def _fix_a_slash_b(string):
    if len(string.split("/")) != 2:
        return string
    a = string.split("/")[0]
    b = string.split("/")[1]
    try:
        a = int(a)
        b = int(b)
        assert string == "{}/{}".format(a, b)
        new_string = "\\frac{" + str(a) + "}{" + str(b) + "}"
        return new_string
    except:
        return string


def _remove_right_units(string):
    # "\\text{ " only ever occurs (at least in the val set) when describing units
    if "\\text{ " in string:
        splits = string.split("\\text{ ")
        assert len(splits) == 2
        return splits[0]
    else:
        return string


def _fix_sqrt(string):
    if "\\sqrt" not in string:
        return string
    splits = string.split("\\sqrt")
    new_string = splits[0] 
    for split in splits[1:]:
        if split[0] != "{":
            a = split[0]
            new_substr = "\\sqrt{" + a + "}" + split[1:]
        else:
            new_substr = "\\sqrt" + split
        new_string += new_substr
    return new_string


def _strip_string(string):
    # linebreaks  
    string = string.replace("\n", "")
    #print(string)

    # remove inverse spaces
    string = string.replace("\\!", "")
    #print(string)

    # replace \\ with \
    string = string.replace("\\\\", "\\")
    #print(string)

    # replace tfrac and dfrac with frac
    string = string.replace("tfrac", "frac")
    string = string.replace("dfrac", "frac")
    #print(string)

    # remove \left and \right
    string = string.replace("\\left", "")
    string = string.replace("\\right", "")
    #print(string)
    
    # Remove circ (degrees)
    string = string.replace("^{\\circ}", "")
    string = string.replace("^\\circ", "")

    # remove dollar signs
    string = string.replace("\\$", "")
    
    # remove units (on the right)
    string = _remove_right_units(string)

    # remove percentage
    string = string.replace("\\%", "")
    string = string.replace("\\%", "")

    # " 0." equivalent to " ." and "{0." equivalent to "{." Alternatively, add "0" if "." is the start of the string
    string = string.replace(" .", " 0.")
    string = string.replace("{.", "{0.")
    # if empty, return empty string
    if len(string) == 0:
        return string
    if string[0] == ".":
        string = "0" + string

    # to consider: get rid of e.g. "k = " or "q = " at beginning
    if len(string.split("=")) == 2:
        if len(string.split("=")[0]) <= 2:
            string = string.split("=")[1]

    # fix sqrt3 --> sqrt{3}
    string = _fix_sqrt(string)

    # remove spaces
    string = string.replace(" ", "")

    # \frac1b or \frac12 --> \frac{1}{b} and \frac{1}{2}, etc. Even works with \frac1{72} (but not \frac{72}1). Also does a/b --> \\frac{a}{b}
    string = _fix_fracs(string)

    # manually change 0.5 --> \frac{1}{2}
    if string == "0.5":
        string = "\\frac{1}{2}"

    # NOTE: X/Y changed to \frac{X}{Y} in dataset, but in simple cases fix in case the model output is X/Y
    string = _fix_a_slash_b(string)

    return string


def last_boxed_only_string(string):
    idx = string.rfind("\\boxed")
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1
    
    if right_brace_idx == None:
        retval = None
    else:
        retval = string[idx:right_brace_idx + 1]
    
    return retval


def remove_boxed(s):
    left = "\\boxed{"
    try:
        assert s[:len(left)] == left
        assert s[-1] == "}"
        return s[len(left):-1]
    except:
        return None
