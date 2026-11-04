"""Prompt slots: a library blueprint keeps its default prompts; a candidate may
replace chosen slots for one run (`reuse` applies an existing blueprint with
new prompts; the library keeps the blueprint, not the task prompts).

    python -m pytest tests/test_prompt_slots.py -q
"""
from __future__ import annotations

import asyncio
import hashlib
import sys
from pathlib import Path

import pytest

from strategist import prompt_slots as P

BLUEPRINT = '''"""Two-step blueprint used by the tests."""
import re

META = {"name": "two_step", "description": "a description that is longer than twenty characters"}
SYSTEM = "You are a careful solver. Answer with the final result only."


async def execute(llm, task: str) -> str:
    """Docstrings are not prompt slots, however long they are."""
    draft = await llm(f"Question:\\n{task}\\n\\nSolve the problem step by step and state the result.", SYSTEM)
    if re.search(r"this pattern is long enough to look like a prompt", draft):
        return draft
    checked = await llm(f"Draft: {draft}\\nCheck the draft and correct any mistake you find. Use {{braces}} freely.", SYSTEM)
    return checked
'''

LIBRARY_BLUEPRINT = '''"""One-call blueprint the tests write into the library."""
from strategist.basic import generate

META = {"name": "zz_test_one_call", "description": "a one-call test blueprint with a template prompt"}


async def execute(llm, task: str) -> str:
    return await generate(llm, f"{task}\\n\\nReason step by step, then give the final answer.", max_tokens=16384)
'''


@pytest.fixture()
def library(tmp_path: Path) -> Path:
    (tmp_path / "two_step.py").write_text(BLUEPRINT)
    return tmp_path


@pytest.fixture()
def library_blueprint():
    """A blueprint placed in the real library for the search-loop test, removed after."""
    from strategist.search import STRATEGIST_WORKFLOWS_DIR

    name = "zz_test_one_call"
    path = STRATEGIST_WORKFLOWS_DIR / f"{name}.py"
    path.write_text(LIBRARY_BLUEPRINT)
    try:
        yield name
    finally:
        path.unlink(missing_ok=True)
        sys.modules.pop(f"strategist.workflows.{name}", None)


def run(fn, task="What is 6*7?"):
    calls = []

    async def llm(prompt, system):
        calls.append((prompt, system))
        return "42"

    return asyncio.run(fn(llm, task)), calls


def test_slots_are_the_prompt_literals_only():
    slots = P.list_prompt_slots(BLUEPRINT)
    assert [s["id"] for s in slots] == ["p1", "p2", "p3"]
    assert slots[0]["default"].startswith("You are a careful solver")          # module-level constant
    assert slots[1]["fstring"] and slots[1]["placeholders"] == ["task"]
    assert slots[2]["placeholders"] == ["draft"] and "{{braces}}" in slots[2]["default"]
    text = " ".join(s["default"] for s in slots)
    assert "description that is longer" not in text       # META is metadata, not a prompt
    assert "Docstrings are not" not in text and "this pattern is long" not in text


def test_defaults_run_when_nothing_is_overridden(library):
    fn, err = P.load_with_overrides("two_step", None, library)
    assert err is None
    out, calls = run(fn)
    assert out == "42" and "Solve the problem step by step" in calls[0][0]


def test_override_replaces_one_slot_and_keeps_the_rest(library):
    before = hashlib.sha256((library / "two_step.py").read_bytes()).hexdigest()
    prompts = {"p2": "Question:\n{task}\n\nThis is an algebra word problem; set up one equation, then solve it."}
    fn, err = P.load_with_overrides("two_step", prompts, library)
    assert err is None
    _out, calls = run(fn)
    assert "set up one equation" in calls[0][0] and "What is 6*7?" in calls[0][0]
    assert "Check the draft" in calls[1][0]                       # p3 keeps its default
    assert calls[0][1].startswith("You are a careful solver")      # p1 keeps its default
    # the library copy is untouched: task prompts stay with the run
    assert hashlib.sha256((library / "two_step.py").read_bytes()).hexdigest() == before
    assert "set up one equation" in P.strategy_source("two_step", prompts, library)
    assert "set up one equation" not in P.strategy_source("two_step", None, library)


def test_plain_text_slot_and_literal_braces(library):
    fn, _ = P.load_with_overrides("two_step", {
        "p1": "Answer like a {strict} grader.",                    # plain slot: braces are literal
        "p3": "Draft: {draft}\nReturn JSON such as {{\"answer\": 1}}.",
    }, library)
    _out, calls = run(fn)
    assert calls[0][1] == "Answer like a {strict} grader."
    assert calls[1][0] == 'Draft: 42\nReturn JSON such as {"answer": 1}.'


@pytest.mark.parametrize("prompts,fragment", [
    ({"p9": "x" * 30}, "unknown prompt slot"),
    ({"p2": "no placeholder here at all"}, "keep exactly the placeholders"),
    ({"p2": "Question: {task} and {secret}"}, "keep exactly the placeholders"),
    ({"p2": "Question: {task} with a stray { brace"}, "not a valid template"),
    ({"p2": "   "}, "non-empty"),
    (["p2"], "must be an object"),
])
def test_bad_overrides_are_reported(library, prompts, fragment):
    problems = P.override_problems(BLUEPRINT, prompts)
    assert problems and fragment in problems[0]
    fn, err = P.load_with_overrides("two_step", prompts, library)
    assert fn is None and err


def test_fingerprint_separates_strategies():
    assert P.overrides_key(None) == "" and P.overrides_key({}) == ""
    a, b = P.overrides_key({"p1": "a" * 30}), P.overrides_key({"p1": "b" * 30})
    assert a and b and a != b
    assert P.overrides_key({"p1": "x", "p2": "y"}) == P.overrides_key({"p2": "y", "p1": "x"})


def test_search_loop_validates_and_resolves_overrides(library_blueprint):
    from strategist.search import STRATEGIST_WORKFLOWS_DIR, resolve_strategy, slate_violations

    name = library_blueprint
    cand = lambda **kw: {"name": name, "kind": "reuse", "tier_mode_anchors": ["multi"],
                         "why": "w", "builds_on": [], **kw}
    slots = P.list_prompt_slots((STRATEGIST_WORKFLOWS_DIR / f"{name}.py").read_text())
    good = {"p1": slots[0]["default"].replace("Reason step by step", "Reason step by step about the algebra")}
    bad = slate_violations([cand(prompts={"p77": "x" * 30})] * 5, [], require_kinds=False)
    assert any("`prompts`" in v and "unknown prompt slot" in v for v in bad)
    assert not any("`prompts`" in v for v in slate_violations([cand(prompts=good)] * 5, [], require_kinds=False))
    fn_default, err1 = resolve_strategy(name, None)
    fn_task, err2 = resolve_strategy(name, good)
    assert err1 is None and err2 is None and fn_default is not fn_task
    text = P.describe_slots(name)
    assert text.startswith("Prompt slots") and "p1 (template, keep {task})" in text


def test_structural_duplicates_ignore_prompts_and_metadata():
    """Blueprints are duplicates when their code matches once the prompts are
    elided. A prompts-only variation is therefore not a new blueprint; it is
    `reuse` plus `prompts`."""
    from strategist.search import structural_hash

    a = ('"""Doc A."""\nMETA = {"name": "a", "traits": {"multi_path": True}}\n\n'
         'async def execute(llm, task):\n    """one"""\n    x = await generate(llm, f"Solve {task} carefully")\n    return x\n')
    b = ('"""Other doc."""\nMETA = {"name": "b", "traits": {"multi_path": False}, "extra": 1}\n\n'
         'async def execute(llm, task):\n    # comment\n    x = await generate(llm,  f"A different prompt for {task}")\n    return x\n')
    assert structural_hash(a) == structural_hash(b)
    assert structural_hash(a) != structural_hash(b.replace("return x", "return x.strip()"))
    assert structural_hash(a) != structural_hash(b.replace("{task}", "{task.upper()}"))
    assert structural_hash("def broken(:") is None
