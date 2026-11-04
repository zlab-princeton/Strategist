"""
Workflow experience store — records the strategist's observations about workflows.

Each time the strategist uses a workflow, it can record:
- What task/benchmark it was used on
- What it observed (strengths, weaknesses, edge cases)
- Its recommendation for future users
- Accuracy impact if known

This builds a growing knowledge base that makes the library smarter over time.
Stored as a single JSON file, searchable and randomly sample-able.
"""

import json
import random
import fcntl
from pathlib import Path
from datetime import datetime

EXPERIENCE_FILE = Path(__file__).parent / "workflow_experience.json"


def _load() -> dict:
    """Read-only load (shared lock). For read-modify-write use `_mutate`
    instead — the split `_load` → mutate → `_save` pattern has a
    lost-update race: between releasing the shared lock and taking the
    exclusive lock, another process can update the file, and our rewrite
    silently drops their change. That race was losing ~80% of
    measurement records under 30-way concurrent writes."""
    try:
        with open(EXPERIENCE_FILE, "r") as f:
            fcntl.flock(f, fcntl.LOCK_SH)
            try:
                data = json.load(f)
            finally:
                fcntl.flock(f, fcntl.LOCK_UN)
        return data
    except FileNotFoundError:
        return {}
    except Exception:
        return {}


def _mutate(fn):
    """Atomic read-modify-write under a single exclusive lock. `fn(data)`
    receives the current dict, mutates it in place, and returns nothing.
    """
    import os as _os
    _os.makedirs(EXPERIENCE_FILE.parent, exist_ok=True)
    # Open r+ so we can both read and rewrite under the same fd+lock.
    # Create an empty JSON object if the file doesn't exist.
    if not EXPERIENCE_FILE.exists():
        EXPERIENCE_FILE.write_text("{}")
    with open(EXPERIENCE_FILE, "r+") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            try:
                data = json.load(f)
            except Exception:
                data = {}
            fn(data)
            f.seek(0)
            f.truncate()
            json.dump(data, f, indent=2)
            f.flush()
            _os.fsync(f.fileno())
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def _save(data: dict):
    """Deprecated; kept for callers that need a full replace. Prefer
    `_mutate` for read-modify-write."""
    with open(EXPERIENCE_FILE, "w") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            json.dump(data, f, indent=2)
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


NARRATIVE_CAP = 20
MEASUREMENT_CAP = 200


def _cap_by_type(entries: list[dict]) -> list[dict]:
    """Keep the last NARRATIVE_CAP narrative records + last MEASUREMENT_CAP
    measurement records. Entries without a `type` field are treated as
    narrative (backward-compat with legacy records)."""
    narrative = [e for e in entries if e.get("type", "narrative") == "narrative"]
    measurement = [e for e in entries if e.get("type") == "measurement"]
    return narrative[-NARRATIVE_CAP:] + measurement[-MEASUREMENT_CAP:]


def record_experience(
    workflow_name: str,
    commander_id: str,
    task_type: str,
    observation: str,
    recommendation: str = "",
    accuracy_impact: str = "",
):
    """Record the strategist's narrative observation about using a workflow."""
    def _apply(data):
        data.setdefault(workflow_name, []).append({
            "type": "narrative",
            "commander": commander_id,
            "task_type": task_type,
            "observation": observation,
            "recommendation": recommendation,
            "accuracy_impact": accuracy_impact,
            "timestamp": datetime.now().isoformat(),
        })
        data[workflow_name] = _cap_by_type(data[workflow_name])
    _mutate(_apply)


def record_measurement(
    workflow_name: str,
    benchmark: str,
    anchor: str,
    feedback_acc: float,
    feedback_cost: float,
    blind_acc: float,
    blind_cost: float,
    hidden_acc: float | None = None,
    hidden_cost: float | None = None,
    iteration: int = 0,
    run_id: str | int | None = None,
    commander: str = "",
    task_type: str = "",
    feedback_n: int | None = None,
    blind_n: int | None = None,
):
    """Record a harness-produced measurement of a workflow on a benchmark
    cell. Distinct from `record_experience` (narrative commentary) —
    measurements are append-only numeric facts from the eval harness."""
    def _apply(data):
        data.setdefault(workflow_name, []).append({
            "type": "measurement",
            "commander": commander,
            "benchmark": benchmark,
            "task_type": task_type,
            "anchor": anchor,
            "feedback_acc": feedback_acc,
            "feedback_cost": feedback_cost,
            "blind_acc": blind_acc,
            "blind_cost": blind_cost,
            "hidden_acc": hidden_acc,
            "hidden_cost": hidden_cost,
            "feedback_n": feedback_n,
            "blind_n": blind_n,
            "iteration": iteration,
            "run_id": run_id,
            "timestamp": datetime.now().isoformat(),
        })
        data[workflow_name] = _cap_by_type(data[workflow_name])
    _mutate(_apply)


def get_experience(workflow_name: str) -> list[dict]:
    """Get all experience notes for a workflow."""
    data = _load()
    return data.get(workflow_name, [])


def search_by_task(task_type: str) -> dict[str, list[dict]]:
    """Find all workflow experiences related to a task type."""
    data = _load()
    results = {}
    task_lower = task_type.lower()
    for wf_name, experiences in data.items():
        matching = [e for e in experiences if task_lower in e.get("task_type", "").lower()]
        if matching:
            results[wf_name] = matching
    return results


def search_by_keyword(keyword: str) -> dict[str, list[dict]]:
    """Search experience notes by keyword."""
    data = _load()
    results = {}
    kw_lower = keyword.lower()
    for wf_name, experiences in data.items():
        matching = [
            e for e in experiences
            if kw_lower in e.get("observation", "").lower()
            or kw_lower in e.get("recommendation", "").lower()
        ]
        if matching:
            results[wf_name] = matching
    return results


def sample_for_inspiration(n: int = 5) -> str:
    """
    Randomly sample workflows + their experience notes.
    This is the AlphaEvolve-style inspiration — show the strategist
    things it wasn't looking for, from different task domains.
    """
    data = _load()
    if not data:
        return "(no experience recorded yet)"

    # Sample workflows that HAVE experience notes
    candidates = [(name, exps) for name, exps in data.items() if exps]
    if not candidates:
        return "(no experience recorded yet)"

    sampled = random.sample(candidates, min(n, len(candidates)))

    lines = []
    for name, exps in sampled:
        # Pick a random experience note
        exp = random.choice(exps)
        lines.append(
            f"- `{name}` (used on {exp.get('task_type', '?')}): "
            f"{exp.get('observation', '')[:120]} "
            f"→ {exp.get('recommendation', '')[:80]}"
        )
    return "\n".join(lines)


def format_for_prompt(workflow_name: str, max_notes: int = 3) -> str:
    """Format experience notes for inclusion in a prompt."""
    exps = get_experience(workflow_name)
    if not exps:
        return ""
    recent = exps[-max_notes:]
    lines = []
    for e in recent:
        lines.append(
            f"    [{e.get('task_type', '?')}] {e.get('observation', '')[:100]}"
            f"{' → ' + e.get('recommendation', '')[:60] if e.get('recommendation') else ''}"
        )
    return "\n".join(lines)
