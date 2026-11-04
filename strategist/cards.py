"""Measurement cards: one markdown card per blueprint in strategist/cards/,
written from its META block and refreshed after every measurement and
experience note."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

STRATEGIST_DIR = Path(__file__).resolve().parent
WORKFLOWS_DIR = STRATEGIST_DIR / "workflows"
CARDS_DIR = STRATEGIST_DIR / "cards"
INDEX_PATH = STRATEGIST_DIR / "index.json"
EXPERIENCE_PATH = STRATEGIST_DIR / "workflow_experience.json"

CARD_TEMPLATE = """# {name}

> **Light card** — written by the search harness from this blueprint's
> measurements and the strategists' experience notes; refreshed after each.

## Summary
`{name}` — {description}

## Designed for
{designed_for}

## When to use
{when_to_use}

## Inputs / output
- Output: {output}
- Uses primitives: {uses}

## Traits
{traits_block}

## Tags
{tags}

## Measured behavior
{measured_block}

## Accumulated experience (notes from earlier searches)
{experience_block}
"""


def _render_measurements(entries: list[dict]) -> str:
    """Render harness measurement records (accuracy always paired with its
    weighted cost) for the card's Measured behavior section."""
    rows = [e for e in entries if e.get("type") == "measurement"]
    if not rows:
        return "(no measured cells yet)"
    out = []
    for e in rows[-8:]:
        bench = e.get("benchmark", "?")
        anchor = e.get("anchor", "?")
        parts = [f"- {bench} @ {anchor}:"]
        for label, acc_key, cost_key, n_key in (("train", "feedback_acc", "feedback_cost", "feedback_n"),
                                                ("val", "blind_acc", "blind_cost", "blind_n")):
            acc = e.get(acc_key)
            if acc is None:
                continue
            cost = float(e.get(cost_key, 0) or 0)
            n = e.get(n_key)
            cost_txt = f"cost/sample={cost / n:.0f}" if n else f"total cost={cost:.0f}"
            parts.append(f"{label} acc={acc:.1%} ({cost_txt})")
        out.append(" ".join(parts))
    return "\n".join(out)


def _load_workflow_meta(path: Path) -> dict:
    """Extract the META dict from a workflow .py file. We don't exec the
    module (imports may fail); we parse literal dicts with ast."""
    import ast

    src = path.read_text()
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return {}
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id == "META"
        ):
            try:
                return ast.literal_eval(node.value)
            except (ValueError, SyntaxError):
                return {}
    return {}


def _render_traits(traits) -> str:
    if not traits:
        return "(none declared)"
    if isinstance(traits, list):
        return "\n".join(f"- **{k}**: yes" for k in traits)
    if not isinstance(traits, dict):
        return "(none declared)"
    rows = []
    for k, v in traits.items():
        if isinstance(v, bool):
            rows.append(f"- **{k}**: {'yes' if v else 'no'}")
        else:
            rows.append(f"- **{k}**: {v}")
    return "\n".join(rows)


def _render_experience(entries: list[dict]) -> str:
    """The strategists' narrative notes (type `narrative`); measurement rows
    live in the Measured behavior section."""
    notes = [e for e in entries if e.get("type", "narrative") == "narrative"
             and (e.get("observation") or e.get("recommendation"))]
    if not notes:
        return "(no notes yet — the strategist that measures this blueprint writes the first one)"
    out = []
    for e in notes[-8:]:  # keep the card short: the latest notes
        obs = (e.get("observation") or "").strip()
        rec = (e.get("recommendation") or "").strip()
        task = e.get("task_type") or "?"
        out.append(f"- [{task}]\n  - observation: {obs}\n  - recommendation: {rec}")
    return "\n".join(out)


def generate_card(workflow_name: str, experience: dict, force: bool = False) -> str | None:
    """Render a light card for one workflow. Returns the path written, or
    None if skipped (existing card and not --force)."""
    card_path = CARDS_DIR / f"{workflow_name}.md"
    if card_path.exists() and not force:
        return None
    wf_path = WORKFLOWS_DIR / f"{workflow_name}.py"
    if not wf_path.exists():
        return None
    meta = _load_workflow_meta(wf_path)
    wf_entries = experience.get(workflow_name, [])

    card_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        from strategist.retrieval import library_meta
        side = library_meta(workflow_name)
    except Exception:
        side = {}
    designed = (f"`{side['designed_for']}` ({side.get('task_kind') or 'task kind unknown'}), written in "
                f"iteration {side.get('iteration', '?')}"
                if side.get("designed_for") else "(a seed blueprint: general purpose)")
    text = CARD_TEMPLATE.format(
        name=workflow_name,
        designed_for=designed,
        description=meta.get("description", "(no description in META)"),
        when_to_use=meta.get("when_to_use", "(no when_to_use in META)"),
        output=meta.get("output", "str"),
        uses=", ".join(meta.get("uses", []) or []) or "(none)",
        traits_block=_render_traits(meta.get("traits", {})),
        tags=", ".join(meta.get("tags", []) or []) or "(none)",
        measured_block=_render_measurements(wf_entries),
        experience_block=_render_experience(wf_entries),
    )
    card_path.write_text(text)
    return str(card_path)


def write_measurement_card(workflow_name: str) -> str | None:
    """Refresh a blueprint's measurement card after evaluation (paper Sec. 3
    step 4: each candidate joins the library with its measurement card and
    experience notes). Light cards are regenerated with the latest measured
    cells; hand-written sweep cards are preserved untouched."""
    card_path = CARDS_DIR / f"{workflow_name}.md"
    experience: dict = {}
    if EXPERIENCE_PATH.exists():
        try:
            experience = json.loads(EXPERIENCE_PATH.read_text())
        except json.JSONDecodeError:
            pass
    if card_path.exists():
        try:
            text = card_path.read_text()
        except OSError:
            text = ""
        if "Light card" not in text[:500]:
            # A seed card (measurement table from the baseline matrix) or a
            # hand-written sweep card: keep it, refresh only its notes section.
            marker = "## Notes from prior use"
            if marker in text:
                head = text.split(marker)[0]
                card_path.write_text(f"{head}{marker}\n{_render_experience(experience.get(workflow_name, []))}\n")
                return str(card_path)
            return None
    return generate_card(workflow_name, experience, force=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--force", action="store_true",
                   help="Overwrite existing light cards (never touches sweep cards)")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    if not CARDS_DIR.exists():
        CARDS_DIR.mkdir(parents=True)

    experience: dict = {}
    if EXPERIENCE_PATH.exists():
        try:
            experience = json.loads(EXPERIENCE_PATH.read_text())
        except json.JSONDecodeError:
            pass

    wf_names = sorted(
        p.stem for p in WORKFLOWS_DIR.glob("*.py")
        if not p.stem.startswith("_")
    )

    # Classify existing cards: "sweep" (full analytical) vs "light".
    sweep_cards: set[str] = set()
    for cp in CARDS_DIR.glob("*.md"):
        content = cp.read_text()[:500]
        if "Light card" not in content:
            sweep_cards.add(cp.stem)

    created = overwritten = skipped = 0
    for name in wf_names:
        if name in sweep_cards and not args.force:
            skipped += 1
            continue
        card_path = CARDS_DIR / f"{name}.md"
        is_update = card_path.exists()
        if args.dry_run:
            action = "would overwrite" if is_update else "would create"
            print(f"  {action}: {card_path.name}")
            if is_update:
                overwritten += 1
            else:
                created += 1
        else:
            result = generate_card(name, experience, force=args.force)
            if result is None:
                skipped += 1
            elif is_update:
                overwritten += 1
            else:
                created += 1

    print()
    print(f"created: {created}")
    print(f"overwritten: {overwritten}")
    print(f"skipped (sweep card preserved): {skipped}")


if __name__ == "__main__":
    main()
