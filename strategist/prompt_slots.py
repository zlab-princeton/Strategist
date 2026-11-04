"""Prompt slots of a blueprint: defaults in the library, overrides per run.

A library blueprint carries its prompts as string literals. Those are the
defaults. A candidate strategy can pair the blueprint with task-specific
prompts by overriding chosen slots for one run (paper Sec. 3, step 2: `reuse`
"directly applies an existing library blueprint with new prompts"). The
library file is never rewritten: the overrides are stored with the run, not in
the library (step 4: the library keeps the blueprint, not the task prompts).

Slots are the prompt-sized string literals of the blueprint, numbered p1, p2,
... in source order. Library files are never overwritten, so the numbering of a
blueprint is stable.

Overriding a plain string replaces it with the new text. Overriding an f-string
takes a template: it must keep exactly the placeholders of the default (for
example `{task}`), and literal braces are doubled, as in any f-string.
"""
from __future__ import annotations

import ast
import hashlib
import json
import types
from pathlib import Path

WORKFLOWS_DIR = Path(__file__).resolve().parent / "workflows"
MIN_PROMPT_CHARS = 20


def _literal_text(node: ast.AST) -> str:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.JoinedStr):
        return "".join(
            v.value for v in node.values if isinstance(v, ast.Constant) and isinstance(v.value, str)
        )
    return ""


def _placeholders(node: ast.AST) -> list[str]:
    if not isinstance(node, ast.JoinedStr):
        return []
    return [ast.unparse(v.value) for v in node.values if isinstance(v, ast.FormattedValue)]


def _slot_nodes(tree: ast.Module) -> list[ast.AST]:
    """String literals that act as prompts, in source order."""
    parent: dict[int, ast.AST] = {}
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            parent[id(child)] = node

    def ancestors(node: ast.AST):
        while id(node) in parent:
            node = parent[id(node)]
            yield node

    docstrings: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node.body:
            first = node.body[0]
            if (isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant)
                    and isinstance(first.value.value, str)):
                docstrings.add(id(first.value))

    def assigned_names(stmt: ast.AST) -> list[str]:
        if isinstance(stmt, ast.Assign):
            return [t.id for t in stmt.targets if isinstance(t, ast.Name)]
        if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
            return [stmt.target.id]
        return []

    slots: list[ast.AST] = []
    for node in ast.walk(tree):
        is_string = isinstance(node, ast.JoinedStr) or (
            isinstance(node, ast.Constant) and isinstance(node.value, str)
        )
        if not is_string or id(node) in docstrings:
            continue
        up = parent.get(id(node))
        if isinstance(up, (ast.JoinedStr, ast.FormattedValue)):
            continue  # a piece of an f-string, or its format spec
        if isinstance(up, ast.Dict) and any(key is node for key in up.keys):
            continue  # dictionary key
        if (isinstance(up, ast.Call) and isinstance(up.func, ast.Attribute)
                and isinstance(up.func.value, ast.Name) and up.func.value.id == "re"):
            continue  # regular expression
        chain = list(ancestors(node))
        if any("META" in assigned_names(a) for a in chain):
            continue  # library metadata
        in_function = any(isinstance(a, (ast.FunctionDef, ast.AsyncFunctionDef)) for a in chain)
        module_constant = any(
            parent.get(id(a)) is tree and any(name.isupper() for name in assigned_names(a)) for a in chain
        )
        if not (in_function or module_constant):
            continue
        if len(_literal_text(node).strip()) < MIN_PROMPT_CHARS:
            continue
        slots.append(node)
    slots.sort(key=lambda n: (n.lineno, n.col_offset))
    return slots


def _template_of(node: ast.AST) -> str:
    """Default text of a slot. For an f-string this is its template, with the
    placeholders in braces and literal braces doubled."""
    if isinstance(node, ast.JoinedStr):
        return ast.literal_eval(ast.unparse(node)[1:])
    return node.value  # type: ignore[union-attr]


def _parse_template(text: str) -> ast.JoinedStr:
    node = ast.parse("f" + repr(text), mode="eval").body
    if not isinstance(node, ast.JoinedStr):
        node = ast.JoinedStr(values=[node])
    return node


def list_prompt_slots(source: str) -> list[dict]:
    """[{id, default, fstring, placeholders, line}] for a blueprint's source."""
    out = []
    for i, node in enumerate(_slot_nodes(ast.parse(source)), 1):
        out.append({
            "id": f"p{i}",
            "default": _template_of(node),
            "fstring": isinstance(node, ast.JoinedStr),
            "placeholders": sorted(set(_placeholders(node))),
            "line": node.lineno,
        })
    return out


def override_problems(source: str, overrides) -> list[str]:
    """Why `overrides` cannot be applied to this blueprint ([] when it can)."""
    if not overrides:
        return []
    if not isinstance(overrides, dict):
        return ["`prompts` must be an object that maps slot ids (p1, p2, ...) to replacement text"]
    nodes = {f"p{i}": n for i, n in enumerate(_slot_nodes(ast.parse(source)), 1)}
    problems: list[str] = []
    for key, text in overrides.items():
        node = nodes.get(key)
        if node is None:
            problems.append(f"unknown prompt slot `{key}` (this blueprint has: {', '.join(nodes) or 'no slots'})")
            continue
        if not isinstance(text, str) or not text.strip():
            problems.append(f"slot `{key}`: the replacement must be non-empty text")
            continue
        if isinstance(node, ast.JoinedStr):
            try:
                new = _parse_template(text)
            except SyntaxError:
                problems.append(f"slot `{key}`: not a valid template (double any literal braces)")
                continue
            want, got = sorted(set(_placeholders(node))), sorted(set(_placeholders(new)))
            if want != got:
                problems.append(
                    f"slot `{key}`: keep exactly the placeholders of the default "
                    f"({', '.join('{' + p + '}' for p in want) or 'none'}); "
                    f"got ({', '.join('{' + p + '}' for p in got) or 'none'})"
                )
    return problems


class _Replace(ast.NodeTransformer):
    def __init__(self, mapping: dict[int, ast.AST]):
        self.mapping = mapping

    def visit(self, node: ast.AST):
        new = self.mapping.get(id(node))
        if new is not None:
            return ast.copy_location(new, node)
        return super().visit(node)


def apply_prompt_overrides(source: str, overrides) -> ast.Module:
    """The blueprint's module with the given slots replaced. Call
    `override_problems` first; unknown slot ids are ignored here."""
    tree = ast.parse(source)
    if not overrides:
        return tree
    mapping: dict[int, ast.AST] = {}
    for i, node in enumerate(_slot_nodes(tree), 1):
        text = overrides.get(f"p{i}")
        if text is None:
            continue
        mapping[id(node)] = (
            _parse_template(text) if isinstance(node, ast.JoinedStr) else ast.Constant(value=text)
        )
    tree = _Replace(mapping).visit(tree)
    ast.fix_missing_locations(tree)
    return tree


def overrides_key(overrides) -> str:
    """Stable fingerprint of a set of overrides ('' for none)."""
    if not overrides:
        return ""
    blob = json.dumps(sorted(overrides.items()), ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def strategy_source(name: str, overrides=None, workflows_dir: Path | None = None) -> str:
    """Source of the strategy = blueprint + overrides (for the run record)."""
    path = (workflows_dir or WORKFLOWS_DIR) / f"{name}.py"
    source = path.read_text()
    return ast.unparse(apply_prompt_overrides(source, overrides)) if overrides else source


def load_with_overrides(name: str, overrides, workflows_dir: Path | None = None):
    """(execute, None) for the blueprint with `overrides` applied, or
    (None, reason). The library file itself is not touched."""
    path = (workflows_dir or WORKFLOWS_DIR) / f"{name}.py"
    try:
        source = path.read_text()
    except OSError as exc:
        return None, f"blueprint file not readable: {exc}"
    try:
        problems = override_problems(source, overrides)
    except SyntaxError as exc:
        return None, f"blueprint does not parse: {exc}"
    if problems:
        return None, "; ".join(problems)
    module = types.ModuleType(f"strategist.workflows.{name}__p{overrides_key(overrides)[:10]}")
    module.__file__ = str(path)
    module.__package__ = "strategist.workflows"
    try:
        exec(compile(apply_prompt_overrides(source, overrides), str(path), "exec"), module.__dict__)
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"
    fn = getattr(module, "execute", None)
    if fn is None or not callable(fn):
        return None, "module has no callable `execute` attribute"
    return fn, None


def describe_slots(name: str, max_slots: int = 8, width: int = 110, workflows_dir: Path | None = None) -> str:
    """Short listing of a blueprint's prompt slots for the strategist prompt."""
    path = (workflows_dir or WORKFLOWS_DIR) / f"{name}.py"
    try:
        slots = list_prompt_slots(path.read_text())
    except (OSError, SyntaxError):
        return ""
    if not slots:
        return ""
    lines = ["Prompt slots (defaults; replace any of them through the candidate's `prompts` field):"]
    for slot in slots[:max_slots]:
        text = " ".join(slot["default"].split())
        if len(text) > width:
            text = text[:width].rstrip() + " ..."
        kind = "template, keep " + ", ".join("{" + p + "}" for p in slot["placeholders"]) if slot["fstring"] and slot["placeholders"] else "text"
        lines.append(f"  - {slot['id']} ({kind}): {text}")
    if len(slots) > max_slots:
        lines.append(f"  - ... and {len(slots) - max_slots} more (p{max_slots + 1} to p{len(slots)})")
    return "\n".join(lines)
