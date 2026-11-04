"""Shared helpers: package paths, JSONL I/O, fenced-code extraction, the
run-history summary, and execution-based grading of code tasks."""
from __future__ import annotations

import json
import os
import random
import re
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parent.parent
STRATEGIST_DIR = Path(__file__).resolve().parent


def load_jsonl(path: str | Path, limit: int | None = None) -> list[dict]:
    samples = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            samples.append(json.loads(line))
    if limit and 0 < limit < len(samples):
        import random
        samples = random.sample(samples, limit)
    return samples


def append_jsonl(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def extract_fenced(text: str, language: str) -> str:
    # Accept fences of 3+ backticks (Claude Code / MiniMax wrapper emits
    # 4-backtick fences around the model's content).
    pattern = re.compile(
        rf"(?ims)^(`{{3,}}){re.escape(language)}[ \t]*\r?\n(.*?)^\1[ \t]*$"
    )
    match = pattern.search(text)
    if not match:
        raise ValueError(f"No ```{language}``` block found in response")
    return match.group(2).strip()


def get_performance_history(log_path: Path, benchmark: str, limit: int = 10) -> str:
    """Read recent performance from evolution log."""
    if not log_path.exists():
        return ""
    lines = []
    for raw in log_path.read_text().splitlines():
        try:
            entry = json.loads(raw)
            if entry.get("benchmark") == benchmark:
                lines.append(entry)
        except json.JSONDecodeError:
            continue

    summaries = []
    for entry in lines[-limit:]:
        s = entry.get("eval_summary", {})
        summaries.append(
            f"  iter={entry.get('iteration')} "
            f"acc={s.get('accuracy', 0):.1%} "
            f"({s.get('passed', 0)}/{s.get('total', 0)}) "
            f"note={entry.get('note', '')[:80]}"
        )
    return "\n".join(summaries) if summaries else ""


# ----------------------------------------------------------------- code tasks

CODE_FENCE_RE = re.compile(r"```(?:python|py)?[ \t]*\n?(.*?)```", re.DOTALL | re.IGNORECASE)


HUMANEVAL_PREFIX = "Complete this Python function:"


ENTRY_POINT_RE = re.compile(r"^\s*(?:async\s+def|def)\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(", re.MULTILINE)


def _extract_code_payload(text: str) -> str:
    body = (text or "").strip("\n")
    match = CODE_FENCE_RE.search(body)
    if match:
        return match.group(1).strip("\n")
    return body


def _humaneval_prompt_source(task: str) -> str:
    body = task.strip()
    if body.startswith(HUMANEVAL_PREFIX):
        body = body[len(HUMANEVAL_PREFIX):].lstrip()
    return body.rstrip() + "\n"


def _humaneval_entry_point(task: str, sample: dict[str, Any]) -> str:
    explicit = str(sample.get("entry_point", "")).strip()
    if explicit:
        return explicit
    match = ENTRY_POINT_RE.search(_humaneval_prompt_source(task))
    return match.group(1) if match else ""


def _normalize_body_block(body: str) -> str:
    lines = body.strip("\n").splitlines()
    nonempty = [line.expandtabs(4) for line in lines if line.strip()]
    if not nonempty:
        return ""
    # A body whose first line lost its indentation (output was strip()-ed)
    # while the rest kept theirs: give the first line the body indentation back.
    indents = [len(line) - len(line.lstrip(" ")) for line in nonempty]
    if len(indents) > 1 and indents[0] == 0 and min(indents[1:]) > 0:
        opener = nonempty[0].rstrip().endswith(":")
        if not opener or indents[1] > min(indents[1:]):
            first = next(i for i, line in enumerate(lines) if line.strip())
            lines[first] = " " * min(indents[1:]) + lines[first].expandtabs(4)
            nonempty = [line.expandtabs(4) for line in lines if line.strip()]
    base_indent = min(len(line) - len(line.lstrip(" ")) for line in nonempty)
    normalized: list[str] = []
    for raw in lines:
        line = raw.expandtabs(4)
        if not line.strip():
            normalized.append("")
            continue
        indent = len(line) - len(line.lstrip(" "))
        normalized.append(line[min(indent, base_indent):])
    return "\n".join(normalized).strip("\n")


def _completion_source(task: str, body: str, entry_point: str = "") -> str:
    """Runnable source for a code completion.

    MBPP-style task (no signature in the prompt): the completion is the source.
    HumanEval-style task: a completion that defines the entry point at column 0
    is a full function and is placed after the prompt, so the prompt's imports
    and helper functions stay available; anything else is a function body and
    is attached to the prompt's signature. A nested helper `def` inside a body
    does not make it a full function."""
    if not task.strip().lower().startswith(HUMANEVAL_PREFIX.lower()):
        return body
    name = entry_point or (ENTRY_POINT_RE.findall(_humaneval_prompt_source(task)) or [""])[-1]
    if name and re.search(rf"^(?:async\s+def|def)\s+{re.escape(name)}\s*\(", body, re.MULTILINE):
        return _humaneval_prompt_source(task) + "    pass\n\n" + body + "\n"
    normalized = _normalize_body_block(body) or "pass"
    return _humaneval_prompt_source(task) + textwrap.indent(normalized, "    ") + "\n"


def _normalize_humaneval_completion(task: str, raw_output: Any) -> str:
    body = _extract_code_payload(str(raw_output or "")).strip("\n")
    if not body:
        return ""
    return _completion_source(task, body)


def _evaluate_humaneval_completion(task: str, sample: dict[str, Any], raw_output: Any, timeout: int = 30) -> tuple[bool, str, str]:
    # timeout raised from 8s to 30s on 2026-04-16: humaneval_plus and
    # mbpp_plus ship exhaustive edge-case test suites (hundreds of
    # assertions per function) that exceed 8s on correct solutions.
    # Symptom: humaneval_plus 4B-thinking-direct scored 0% with 134/164
    # samples marked `timeout_after_8s` while the model's code was
    # correct. 30s covers the observed tail with ~3× headroom.
    body = _extract_code_payload(str(raw_output or "")).strip("\n")
    if not body:
        return False, "", "empty_completion"
    entry_point = _humaneval_entry_point(task, sample)
    source = _completion_source(task, body, entry_point)
    if not entry_point:
        return False, source, "missing_entry_point"

    test_code = str(sample.get("test_code", "")).strip()
    if not test_code:
        return False, source, "missing_test_code"

    # mbpp_plus test_code runs assertions inline (no `def check()` wrapper);
    # mbpp/humaneval/humaneval_plus define `def check(candidate)` and need an
    # explicit invocation. Only append the call when check() is actually defined.
    if "def check(" in test_code:
        harness = (
            source
            + "\n\n"
            + test_code
            + "\n\n"
            + f"check({entry_point})\n"
            + "print('PASS')\n"
        )
    else:
        harness = (
            source
            + "\n\n"
            + test_code
            + "\n\n"
            + "print('PASS')\n"
        )
    try:
        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False, encoding="utf-8") as handle:
            handle.write(harness)
            script_path = handle.name
        # Run the tests with this interpreter, not whichever `python3` is first
        # on PATH: an older system Python (3.9) rejects EvalPlus test files whose
        # long lines contain non-ASCII text ("Non-UTF-8 code ...").
        result = subprocess.run(
            [sys.executable, script_path],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return False, source, f"timeout_after_{timeout}s"
    except Exception as exc:
        return False, source, f"exec_failure:{exc}"
    finally:
        try:
            os.unlink(script_path)
        except Exception:
            pass

    if result.returncode == 0:
        return True, source, ""

    failure = (result.stderr or result.stdout or "").strip().splitlines()
    note = failure[-1][:240] if failure else "unknown_test_failure"
    return False, source, note
