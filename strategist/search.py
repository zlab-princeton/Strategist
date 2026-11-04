"""
The Strategist search loop (paper Section 3, Algorithm 1).

Contract per iteration:
  1. The strategist LLM emits
     5 candidate strategies plus a JSON slate with `kind`, `tier_mode_anchors`,
     `search_queries`, `read_cards`, `experience`. Slates that omit required
     fields, miss a kind (new/composed/reuse), or structurally duplicate a
     recent candidate are rejected and the strategist is re-prompted.
  2. The harness runs every candidate on the training (feedback) + validation
     (blind) + hidden splits — 100 samples each, capped — recording per-call
     cost breakdown and traces. The winner across all iterations is the
     candidate with the highest hidden-set accuracy (ties: lower cost); the
     final number is reported on the deterministic 200-sample test split.
  3. Training traces + validation aggregates are injected into the next
     iteration's prompt; every candidate is accumulated into the library with
     a measurement card and the strategist's experience notes (AST-level
     structural dedup, task prompts irrelevant to identity).

Terminology map (code → paper): feedback → training, blind → validation,
hidden → hidden, test → test.

Multi-tier `llms` dict (always passed, always 6 entries):
  {
    "fast":     {"nothink": <client>, "thinking": <client>},
    "balanced": {"nothink": <client>, "thinking": <client>},
    "complex":  {"nothink": <client>, "thinking": <client>},
  }

Each client is an openai.AsyncOpenAI augmented with `_bp_model_name` and
`_bp_enable_thinking` so generate/chat/agent route correctly. `single_tier`
test flag: all 6 entries redirect to one anchor.

Workflow signature auto-detect:
  - async def foo(llm,  q) -> str   → single-client (anchor-resolved)
  - async def foo(llms, q) -> str   → full 6-entry dict
"""

from __future__ import annotations

import argparse
import ast
import asyncio
import contextvars
import inspect
import json
import os
import random
import re
import sys
import time
import traceback
import types
from pathlib import Path
from typing import Any, Callable

import openai

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from strategist.backends import (
    CodexSessionLost,
    claude_minimax_call,
    claude_opus_call,
    codex_call,
)
from strategist.common import (
    STRATEGIST_DIR,
    _evaluate_humaneval_completion,
    append_jsonl,
    extract_fenced,
    get_performance_history,
    load_jsonl,
)


def load_jsonl_deterministic(
    path: str | Path, limit: int | None = None, seed: int | None = None
) -> list[dict]:
    """Read JSONL from `path`.

    - `seed=None`: returns rows in file order (used for the test set — same
      rows every run for a given file → reproducible final scores).
    - `seed=<int>`: deterministically shuffles rows using that seed, so
      different seeds pick different subsets while same seed + same file
      always returns the same list. Used by sweep runs where each run-id
      gets its own feedback/blind split.
    """
    rows: list[dict] = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    if seed is not None:
        rng = random.Random(seed)
        rng.shuffle(rows)
    if limit and 0 < limit < len(rows):
        rows = rows[:limit]
    return rows

COMMANDERS = {
    "claude-opus": claude_opus_call,
    "codex": codex_call,
    "claude-minimax": claude_minimax_call,
}
from strategist import paired_pick
from strategist.prompt_slots import (
    describe_slots,
    load_with_overrides,
    override_problems,
    overrides_key,
    strategy_source,
)

RUN_DIR = STRATEGIST_DIR.parent / "runs"
RUN_RAW_DIR = RUN_DIR / "raw"
RUN_PROMPTS_DIR = RUN_DIR / "prompts_sent"
RUN_LOG = RUN_DIR / "log.jsonl"


def _use_out_dir(path: Path) -> None:
    """Repoint the module-level output paths to a per-run directory. Called
    once at startup when --out-dir is passed so concurrent sweep runs don't
    collide."""
    global RUN_DIR, RUN_RAW_DIR, RUN_PROMPTS_DIR, RUN_LOG
    RUN_DIR = path
    RUN_RAW_DIR = RUN_DIR / "raw"
    RUN_PROMPTS_DIR = RUN_DIR / "prompts_sent"
    RUN_LOG = RUN_DIR / "log.jsonl"


def _ensure_out_dirs() -> None:
    for d in (RUN_DIR, RUN_RAW_DIR, RUN_PROMPTS_DIR):
        d.mkdir(parents=True, exist_ok=True)

STRATEGIST_WORKFLOWS_DIR = STRATEGIST_DIR / "workflows"


def _workflow_content_hash(name: str) -> str | None:
    """SHA256 of the library workflow file. Returns None if the file is
    missing or unreadable. Used as the cache key so an in-place file
    change invalidates the cached eval result."""
    import hashlib
    p = STRATEGIST_WORKFLOWS_DIR / f"{name}.py"
    try:
        return hashlib.sha256(p.read_bytes()).hexdigest()
    except OSError:
        return None


def _eval_cache_path() -> Path:
    """Per-run cache file. Lives inside the out-dir so runs don't pollute
    each other (feedback/blind sample sets differ by run_seed, so caching
    across runs would be unsound without also keying on the sample fingerprint)."""
    return RUN_DIR / "eval_cache.jsonl"


def _load_eval_cache() -> dict[tuple[str, str], dict]:
    """Return {(workflow_hash, anchor): cached_summaries} from the per-run
    cache file. One record per line. Later entries override earlier ones
    under the same key — so re-running a cell updates the cache."""
    cache: dict[tuple[str, str], dict] = {}
    path = _eval_cache_path()
    if not path.exists():
        return cache
    try:
        for line in path.open():
            try:
                rec = json.loads(line)
            except Exception:
                continue
            key = (rec.get("workflow_hash"), rec.get("anchor"))
            if key[0] and key[1]:
                cache[key] = rec
    except Exception:
        return {}
    return cache


def _append_eval_cache(rec: dict) -> None:
    try:
        _eval_cache_path().parent.mkdir(parents=True, exist_ok=True)
        with _eval_cache_path().open("a") as f:
            f.write(json.dumps(rec) + "\n")
    except Exception as e:
        print(f"    [cache] write failed: {e}")

# Per-token cost weights proportional to parameter count (paper Table 4):
# Qwen3-0.6B : 1.7B : 4B = 1.00 : 2.83 : 6.67.
TIER_WEIGHTS = {"fast": 1.00, "balanced": 2.83, "complex": 6.67}
TIERS = ("fast", "balanced", "complex")
# Characters of each call's output kept in the per-sample trace, and how much
# of it is shown per call in the strategist's feedback on failed samples.
TRACE_TAIL_CHARS = 600
# Every executor call is kept in full (request and response) and written by
# eval_candidate to <out-dir>/generations/...; set STRATEGIST_SAVE_GENERATIONS=0
# to keep only token counts and the trace tail.
SAVE_GENERATIONS = os.environ.get("STRATEGIST_SAVE_GENERATIONS", "1") != "0"
_HEAVY_RECORD_KEYS = ("request", "response", "transcript")
TRACE_SHOWN_CHARS = 160
MODES = ("nothink", "thinking")

# Naming rules for new library workflows (enforced in ingest_new_workflows).
_NAME_RE = re.compile(r"^[a-z][a-z0-9_]*$")
_BANNED_PREFIXES = ("iter_", "candidate_", "tmp_", "test_")

# ---------------------------------------------------------------------------
# Structural deduplication (paper A.4). Two blueprints are duplicates when
# their ASTs hash to the same value after task-specific string literals
# (prompts, format instructions) are elided; whitespace and comments never
# matter because the hash is over the AST dump, not the source text.
# ---------------------------------------------------------------------------

_LIB_HASH_INDEX = STRATEGIST_DIR / "library_hashes.json"


def structural_hash(source: str) -> str | None:
    """SHA256 over the code of `source` (metadata and docstrings left out)
    with every string constant blanked. Returns None when it does not parse."""
    import hashlib

    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None

    def is_docstring(stmt: ast.stmt) -> bool:
        return (isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant)
                and isinstance(stmt.value.value, str))

    def is_meta(stmt: ast.stmt) -> bool:
        return isinstance(stmt, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "META" for target in stmt.targets
        )

    # Paper A.4 hashes the `execute` function body. The hash therefore covers
    # the code only: `execute` and the helpers and constants it uses. The META
    # block and docstrings describe a blueprint and are not part of its
    # structure, so two files that differ only there are the same blueprint.
    tree.body = [stmt for stmt in tree.body if not is_meta(stmt)]
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if node.body and is_docstring(node.body[0]):
                node.body = node.body[1:] or [ast.Pass()]
    for node in ast.walk(tree):
        if isinstance(node, ast.JoinedStr):
            # An f-string prompt is reduced to its placeholders: how much literal
            # text sits around them is prompt wording, not structure.
            node.values = [v for v in node.values if isinstance(v, ast.FormattedValue)]
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            node.value = ""
    dumped = ast.dump(tree, annotate_fields=False, include_attributes=False)
    return hashlib.sha256(dumped.encode()).hexdigest()


def _load_library_hashes() -> dict[str, str]:
    """Return {structural_hash: workflow_name} for the library. Built lazily
    on first use and persisted; incremental entries are appended by
    `_register_library_hash`. A stale index only weakens dedup (never blocks a
    legitimate write), so corruption falls back to a rebuild."""
    if _LIB_HASH_INDEX.exists():
        try:
            data = json.loads(_LIB_HASH_INDEX.read_text())
            if isinstance(data, dict) and data:
                return data
        except Exception:
            pass
    index: dict[str, str] = {}
    for path in STRATEGIST_WORKFLOWS_DIR.glob("*.py"):
        if path.stem.startswith("_"):
            continue
        try:
            h = structural_hash(path.read_text())
        except OSError:
            continue
        if h and h not in index:
            index[h] = path.stem
    try:
        _LIB_HASH_INDEX.write_text(json.dumps(index))
    except OSError:
        pass
    return index


def _register_library_hash(h: str, name: str) -> None:
    import fcntl

    try:
        if not _LIB_HASH_INDEX.exists():
            _LIB_HASH_INDEX.write_text("{}")
        with open(_LIB_HASH_INDEX, "r+") as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            try:
                try:
                    data = json.load(f)
                except Exception:
                    data = {}
                data[h] = name
                f.seek(0)
                f.truncate()
                json.dump(data, f)
            finally:
                fcntl.flock(f, fcntl.LOCK_UN)
    except OSError:
        pass


def ingest_new_workflows(raw_response: str) -> tuple[list[str], list[str]]:
    """Parse the commander response for `# WORKFLOW: <name>` fenced blocks
    and write each one to strategist/workflows/<name>.py. Runs BEFORE slate
    resolution so any name the commander declared can be imported downstream.

    Returns (written_names, warnings). Existing library files are never
    overwritten — if a commander-proposed name collides with an existing
    library entry, we skip the write and emit a warning so the audit can
    downgrade the slate entry from `new`/`composed` to `reuse`.
    """
    # Match either: marker outside the fence, or marker as first line inside.
    patterns = (
        re.compile(r"#\s*WORKFLOW:\s*([a-zA-Z0-9_]+)\s*\n```python\s*\n(.*?)```", re.DOTALL),
        re.compile(r"```python\s*\n#\s*WORKFLOW:\s*([a-zA-Z0-9_]+)\s*\n(.*?)```", re.DOTALL),
    )
    seen: set[str] = set()
    written: list[str] = []
    warnings: list[str] = []
    for pat in patterns:
        for match in pat.finditer(raw_response):
            name = match.group(1).strip()
            body = match.group(2).strip()
            if name in seen:
                continue
            seen.add(name)

            if not _NAME_RE.match(name):
                warnings.append(
                    f"`# WORKFLOW: {name}` rejected: name must match "
                    f"`^[a-z][a-z0-9_]*$` (snake_case, starts with lowercase)"
                )
                continue
            if any(name.startswith(p) for p in _BANNED_PREFIXES):
                warnings.append(
                    f"`# WORKFLOW: {name}` rejected: name starts with a banned "
                    f"prefix ({', '.join(_BANNED_PREFIXES)}) — pick a strategy-descriptive name"
                )
                continue

            # Strip `from __future__` lines — ast.parse rejects them mid-body
            # and we don't need future imports in library modules.
            body = re.sub(
                r"^\s*from\s+__future__\s+import\s+[^\n]*\n",
                "", body, flags=re.MULTILINE,
            )
            try:
                ast.parse(body)
            except SyntaxError as e:
                warnings.append(
                    f"`# WORKFLOW: {name}` rejected: SyntaxError ({e.msg} at line {e.lineno})"
                )
                continue

            # Anti-cheat: reject workflows that locally execute task-derived
            # code to compute the answer. cruxeval is the canonical vulnerability
            # — the prompt contains the full python function and can be exec()ed
            # for free 100%. Patterns that indicate this:
            #   - calling subprocess on python -c with task content
            #   - exec/eval on a string built from the task
            #   - importing builtins via __import__ (used to sandbox-escape)
            # Match conservatively: bare `exec(` or `eval(` or `subprocess.`
            # appearing as a standalone call. `re.compile` and `ast.literal_eval`
            # are legitimate and pass.
            cheat_patterns = [
                ("subprocess.", r"\bsubprocess\.(run|Popen|call|check_output)\b"),
                ("os.system",   r"\bos\.system\b"),
                ("os.popen",    r"\bos\.popen\b"),
                ("__import__",  r"\b__import__\s*\("),
                ("exec()",      r"(?<![a-zA-Z_])exec\s*\("),
                ("eval()",      r"(?<![a-zA-Z_.])eval\s*\("),
            ]
            cheat_hits = []
            for label, pat in cheat_patterns:
                if re.search(pat, body):
                    cheat_hits.append(label)
            if cheat_hits:
                warnings.append(
                    f"`# WORKFLOW: {name}` rejected: contains forbidden "
                    f"local-execution pattern(s) {cheat_hits}. "
                    f"Workflows must produce the answer through LLM reasoning, "
                    f"not by exec/subprocess on task-derived code."
                )
                continue

            dest = STRATEGIST_WORKFLOWS_DIR / f"{name}.py"
            if dest.exists():
                warnings.append(
                    f"`# WORKFLOW: {name}` collides with existing library file — "
                    f"not overwritten; treat this name as `kind:\"reuse\"` in the slate"
                )
                continue

            # Structural dedup (paper A.4): reject a blueprint whose AST —
            # with string literals elided — matches an existing library entry.
            # Prompt-only variations are not new blueprints.
            h = structural_hash(body)
            if h:
                dup_of = _load_library_hashes().get(h)
                if dup_of and dup_of != name:
                    warnings.append(
                        f"`# WORKFLOW: {name}` rejected: structural duplicate of "
                        f"library blueprint `{dup_of}` (identical AST once prompts "
                        f"are elided) — reuse `{dup_of}` with new prompts instead"
                    )
                    continue

            try:
                dest.write_text(body + ("\n" if not body.endswith("\n") else ""))
                written.append(name)
                if h:
                    _register_library_hash(h, name)
            except OSError as e:
                warnings.append(f"`# WORKFLOW: {name}` write failed: {e}")
    return written, warnings

# Cost-record contextvar. Every traced call pushes a record here; eval_one
# drains it per sample.
_cost_var: contextvars.ContextVar[list[dict]] = contextvars.ContextVar(
    "_strategist_cost", default=None
)


# ---------------------------------------------------------------------------
# Client construction
# ---------------------------------------------------------------------------


class _TierPool:
    """Every live replica of a tier's model. Each call goes to the replica with
    the fewest calls in flight, so one run spreads over all servers of the tier
    instead of binding to the one endpoint it was started with (which left
    replicas idle while others queued hundreds of requests).
    Replicas come from the per-model registry kept by the fleet keeper
    (strategist/logs/vllm_endpoints_<model>.json), re-read every 60 s so
    servers added or removed later are followed. The endpoint given on the
    command line is used only until the registry is read: a server's model
    can change (a fleet can be re-laid out while runs are on,
    and the still-trusted start endpoint then answered 404 "model does not
    exist" for the rest of 13 runs), so only registry-listed replicas serve,
    and a replica that answers 404 is set aside at once."""

    def __init__(self, model_id: str, seed_url: str):
        self.model_id = model_id
        self.seed = str(seed_url).rstrip("/")
        self.clients: dict[str, openai.AsyncOpenAI] = {}
        self.inflight: dict[str, int] = {}
        self.down_until: dict[str, float] = {}
        self.listed: dict[str, bool] = {}
        self.refreshed = 0.0
        self._add(self.seed)
        self.refresh(force=True)

    def _add(self, url: str) -> None:
        if url not in self.clients:
            c = openai.AsyncOpenAI(base_url=url, api_key="EMPTY", timeout=3600.0)
            c._bp_model_name = self.model_id
            self.clients[url] = c
            self.inflight[url] = 0
            self.down_until[url] = 0.0
            self.listed[url] = True

    def refresh(self, force: bool = False) -> None:
        if not force and time.time() - self.refreshed < 60:
            return
        self.refreshed = time.time()
        try:
            urls = [str(u).rstrip("/") for u in json.loads(_registry_path_for_model(self.model_id).read_text())]
        except Exception:
            return
        if not urls:
            return
        for url in urls:
            self._add(url)
        for url in self.clients:
            self.listed[url] = url in urls

    def live_urls(self) -> list[str]:
        self.refresh()
        now = time.time()
        live = [u for u in self.clients if self.listed[u] and self.down_until[u] <= now]
        return live or [u for u in self.clients if self.listed[u]] or list(self.clients)

    def pick(self) -> tuple[str, openai.AsyncOpenAI]:
        urls = self.live_urls()
        random.shuffle(urls)
        url = min(urls, key=lambda u: self.inflight[u])
        return url, self.clients[url]

    def mark_down(self, url: str, seconds: float = 30.0) -> None:
        self.down_until[url] = time.time() + seconds

    def wrong_model(self, url: str) -> None:
        """404 from a replica: it no longer serves this model. Set it aside for
        a long while and re-read the registry on the next pick."""
        self.down_until[url] = time.time() + 600
        self.refreshed = 0.0


def _wrap_cost(client: openai.AsyncOpenAI, tier: str, mode: str,
               pool: "_TierPool | None" = None) -> types.SimpleNamespace:
    """Return a traced client: same API shape, records every create() call
    into the current contextvar's cost list (if set).

    With a `pool`, every call picks the least-loaded live replica of the tier.
    Also: retries up to 3 times on APIConnectionError / Timeout. If the bound
    endpoint dies mid-iteration, this picks a live replacement (the pool, or
    the per-model registry) and swaps in-place for subsequent calls. This
    prevents the 3500+ APIConnectionError flood we saw in the 30-bench sweep."""
    # Use a mutable container so swap-in persists across calls to traced_create.
    client_box = {"c": client}
    model_id = getattr(client, "_bp_model_name", None)

    async def traced_create(*args, **kwargs):
        start = time.perf_counter()
        started_at = time.strftime("%Y-%m-%dT%H:%M:%S")
        response = None
        last_exc = None
        used_url = ""
        # Transient errors we retry: connection drop, timeout, 5xx.
        for attempt in range(3):
            url = None
            if pool is not None:
                url, chosen = pool.pick()
                client_box["c"] = chosen
                pool.inflight[url] += 1
            target = client_box["c"]          # the client this attempt really uses
            used_url = str(getattr(target, "base_url", "") or "")
            try:
                response = await target.chat.completions.create(*args, **kwargs)
                break
            except openai.NotFoundError as exc:
                last_exc = exc
                if pool is None or url is None:
                    raise
                pool.wrong_model(url)         # the next attempt picks another replica
                await asyncio.sleep(0.5 * (attempt + 1))
            except (openai.APIConnectionError, openai.APITimeoutError,
                    openai.InternalServerError) as exc:
                last_exc = exc
                if pool is not None and url is not None:
                    pool.mark_down(url)       # the next attempt picks another replica
                elif model_id:
                    # Try swapping to a live endpoint (prefer the current one if
                    # it recovers, else any live one from the registry).
                    current_ep = str(getattr(client_box["c"], "base_url", "")).rstrip("/")
                    new_ep = _pick_live_endpoint(model_id, prefer=current_ep)
                    if new_ep and new_ep != current_ep:
                        fresh = openai.AsyncOpenAI(base_url=new_ep, api_key="EMPTY", timeout=3600.0)
                        fresh._bp_model_name = model_id
                        fresh._bp_enable_thinking = getattr(client_box["c"], "_bp_enable_thinking", None)
                        client_box["c"] = fresh
                # Brief backoff, then retry.
                await asyncio.sleep(0.5 * (attempt + 1))
            except Exception:
                raise
            finally:
                if pool is not None and url is not None:
                    pool.inflight[url] -= 1
        if response is None:
            raise last_exc  # type: ignore[misc]

        elapsed = time.perf_counter() - start
        usage = getattr(response, "usage", None)
        in_tok = int(getattr(usage, "prompt_tokens", 0) or 0)
        out_tok = int(getattr(usage, "completion_tokens", 0) or 0)
        # Reasoning trace (paper Sec. 3, step 3): keep the end of what this
        # call produced, so the strategist can read how a failed sample went.
        out_tail = ""
        try:
            message = response.choices[0].message
            out_tail = str(getattr(message, "content", None)
                           or getattr(message, "reasoning_content", None) or "")[-TRACE_TAIL_CHARS:]
        except Exception:
            pass
        records = _cost_var.get()
        if records is not None:
            record = {
                "tier": tier,
                "mode": mode,
                "in_tok": in_tok,
                "out_tok": out_tok,
                "latency": round(elapsed, 2),
                "out_tail": out_tail,
            }
            if SAVE_GENERATIONS:
                choices = []
                for ch in getattr(response, "choices", None) or []:
                    msg = getattr(ch, "message", None)
                    tool_calls = getattr(msg, "tool_calls", None) if msg is not None else None
                    choices.append({
                        "content": getattr(msg, "content", None) if msg is not None else None,
                        "reasoning_content": getattr(msg, "reasoning_content", None) if msg is not None else None,
                        "finish_reason": getattr(ch, "finish_reason", None),
                        "tool_calls": [tc.model_dump() for tc in tool_calls] if tool_calls else None,
                    })
                record.update({
                    "source": "chat",
                    "endpoint": used_url,
                    "model": kwargs.get("model"),
                    "started_at": started_at,
                    "request": dict(kwargs),
                    "response": {"id": getattr(response, "id", None), "choices": choices},
                })
            records.append(record)
        return response

    # Build a namespace that looks like AsyncOpenAI to the workflow, but
    # passes through all other attrs so agent.py can still see base_url etc.
    # Use property-style getters so base_url / _bp_model_name reflect the
    # CURRENT (possibly swapped) endpoint.
    class _ClientView(types.SimpleNamespace):
        def __getattribute__(self, name):
            if name == "base_url":
                return getattr(client_box["c"], "base_url", None)
            return super().__getattribute__(name)
    wrapper = _ClientView(
        chat=types.SimpleNamespace(
            completions=types.SimpleNamespace(create=traced_create)
        ),
        _bp_model_name=model_id,
        _bp_enable_thinking=getattr(client, "_bp_enable_thinking", None),
        _bp_provider=getattr(client, "_bp_provider", "vllm"),
        _bp_reasoning_effort=getattr(client, "_bp_reasoning_effort", None),
        _bp_tier=tier,
        _bp_mode=mode,
        _bp_pool=pool,
    )
    if pool is not None:
        wrapper._bp_endpoints_getter = pool.live_urls   # agent() spreads its calls too

    # Compat shim: library workflows that call `await llm.generate(...)`
    # method-style (e.g. principle_eliminate_nothink, cot_backtrack) used to
    # crash 100/100 samples here with AttributeError, which polluted the
    # commander's experience with false negatives. Bind the free functions
    # to this instance so both styles work.
    from strategist.basic import generate as _bp_generate, agent as _bp_agent

    async def _gen(*a, **kw):
        return await _bp_generate(wrapper, *a, **kw)

    async def _agt(*a, **kw):
        return await _bp_agent(wrapper, *a, **kw)

    wrapper.generate = _gen
    wrapper.agent = _agt
    return wrapper


def build_llms_dict(
    endpoints: dict[str, str], model_names: dict[str, str]
) -> dict[str, dict[str, types.SimpleNamespace]]:
    """Build the 6-entry `llms` dict. `endpoints[tier]` is a vLLM base URL;
    `model_names[tier]` is the model id to send in the completion request."""
    llms: dict[str, dict[str, types.SimpleNamespace]] = {}
    for tier in TIERS:
        if "thinking" in str(model_names.get(tier, "")).lower():
            # e.g. Qwen/Qwen3-4B-Thinking-2507: its chat template has no
            # enable_thinking switch, so nothink requests still think.
            print(f"[warn] {tier} tier model {model_names[tier]} is a thinking-only checkpoint: "
                  f"`{tier}/nothink` calls will run in thinking mode. The paper's six tier/mode "
                  f"configurations need a model with both modes (Qwen/Qwen3-4B has them).",
                  flush=True)
    for tier in TIERS:
        base_url = endpoints[tier]
        model = model_names[tier]
        # All replicas of the tier's model, least-loaded first; the given
        # endpoint is one of them (timeout=3600: complex/thinking + long
        # max_tokens under contention can exceed the default 600s).
        pool = _TierPool(model, base_url)
        tier_dict: dict[str, types.SimpleNamespace] = {}
        for mode in MODES:
            raw = openai.AsyncOpenAI(base_url=base_url, api_key="EMPTY", timeout=3600.0)
            raw._bp_model_name = model
            raw._bp_enable_thinking = (mode == "thinking")
            tier_dict[mode] = _wrap_cost(raw, tier, mode, pool=pool)
        llms[tier] = tier_dict
    return llms


def _registry_path_for_model(model_id: str) -> Path:
    """Map 'Qwen/Qwen3-0.6B' → strategist/logs/vllm_endpoints_qwen_qwen3_0_6b.json."""
    key = model_id.lower().replace("/", "_").replace("-", "_").replace(".", "_")
    return STRATEGIST_DIR / "logs" / f"vllm_endpoints_{key}.json"


def _probe_endpoint(base_url: str, timeout: float = 3.0) -> bool:
    """Return True if this vLLM endpoint responds to /models."""
    import subprocess
    try:
        r = subprocess.run(
            ["curl", "-s", "--max-time", str(timeout), base_url.rstrip("/") + "/models"],
            capture_output=True, text=True, timeout=timeout + 1,
        )
        return "model" in (r.stdout or "")
    except Exception:
        return False


def _pick_live_endpoint(model_id: str, prefer: str | None = None) -> str | None:
    """Read the per-model registry and return the first endpoint that answers.
    If `prefer` is supplied and still live, prefer it. Returns None if all dead."""
    reg = _registry_path_for_model(model_id)
    if not reg.exists():
        return None
    try:
        endpoints = json.load(open(reg))
    except Exception:
        return None
    if prefer and prefer in endpoints and _probe_endpoint(prefer):
        return prefer
    for ep in endpoints:
        if _probe_endpoint(ep):
            return ep
    return None


def refresh_llms_dict(
    llms: dict[str, dict[str, types.SimpleNamespace]],
    model_names: dict[str, str],
) -> dict[str, dict[str, types.SimpleNamespace]]:
    """Replace any tier whose current base_url is unreachable with a live
    endpoint from the registry. Returns the same llms dict, mutated in place.
    Logs per-tier status to stdout so failures are visible in slurm logs."""
    for tier in TIERS:
        pool = getattr(llms[tier]["nothink"], "_bp_pool", None)
        if pool is not None:
            pool.refresh(force=True)
            live = pool.live_urls()
            print(f"[endpoint-refresh] {tier}: {len(live)} replica(s) in the pool")
            continue
        current = str(llms[tier]["nothink"].base_url).rstrip("/")
        if current.endswith("/v1"):
            probe_base = current
        else:
            probe_base = current  # probe whatever base_url the wrapper holds
        if _probe_endpoint(probe_base):
            print(f"[endpoint-refresh] {tier}: {current} OK (kept)")
            continue
        live = _pick_live_endpoint(model_names[tier], prefer=None)
        if live is None:
            print(f"[endpoint-refresh] {tier}: {current} DEAD — no live replacement found")
            continue
        print(f"[endpoint-refresh] {tier}: {current} DEAD → {live}")
        model = model_names[tier]
        new_tier: dict[str, types.SimpleNamespace] = {}
        for mode in MODES:
            raw = openai.AsyncOpenAI(base_url=live, api_key="EMPTY", timeout=3600.0)
            raw._bp_model_name = model
            raw._bp_enable_thinking = (mode == "thinking")
            new_tier[mode] = _wrap_cost(raw, tier, mode)
        llms[tier] = new_tier
    return llms


def redirect_all_to(
    llms: dict[str, dict[str, types.SimpleNamespace]], anchor: str
) -> dict[str, dict[str, types.SimpleNamespace]]:
    """For a single-tier test: replace all 6 entries with the one anchor's
    client. anchor format: "{tier}/{mode}"."""
    tier, mode = anchor.split("/")
    pinned = llms[tier][mode]
    out: dict[str, dict[str, types.SimpleNamespace]] = {}
    for t in TIERS:
        out[t] = {m: pinned for m in MODES}
    return out


# ---------------------------------------------------------------------------
# Slate parsing + library resolution
# ---------------------------------------------------------------------------


VALID_KINDS = {"new", "composed", "reuse"}
# What a candidate is designed to do against the top baseline cell: push
# accuracy above it whatever the cost, hold its accuracy at lower cost, or
# land near it at a fraction of the cost. Every slate needs one of each.
VALID_TARGETS = {"accuracy", "efficiency", "balanced"}

# Appended to the strategist prompt in --no-library runs (paper Sec. 7 ablations).
NO_LIBRARY_NOTE = (
    "\n\n---\n\n## Library disabled for this run\n"
    "The blueprint library is switched off. Do not import, reuse or compose library "
    "blueprints; you may only build on blueprints you wrote in earlier iterations of "
    "this run, and on the `strategist.workflows._qa_common` helpers. The "
    "one-of-each-kind rule is waived: all five candidates may be `new`."
)

EMPTY_LIBRARY_NOTE = (
    "\n\n---\n\n## The library is empty\n"
    "No blueprint has been written yet, so there is nothing to reuse or compose this "
    "iteration: all five candidates may be `new`. From the next iteration on, the "
    "blueprints you write here are in the library and the one-of-each-kind rule applies."
)


def library_size() -> int:
    """Blueprints in the library (helper modules starting with `_` do not count)."""
    return sum(1 for f in STRATEGIST_WORKFLOWS_DIR.glob("*.py") if not f.name.startswith("_"))


def parse_slate(raw_response: str) -> dict:
    """Extract and parse the JSON slate block from the commander response.
    Raises ValueError if the block is missing or unparseable. The library
    ingest step (writing new workflow files) is done separately via
    `ingest_new_workflows`; this function only parses metadata.
    """
    try:
        json_block = extract_fenced(raw_response, "json")
    except ValueError as e:
        raise ValueError(f"no fenced json slate block in response: {e}")
    try:
        meta = json.loads(json_block)
    except json.JSONDecodeError as e:
        raise ValueError(f"json slate block failed to parse: {e}")
    if not isinstance(meta, dict):
        raise ValueError(f"json slate block must be an object, got {type(meta).__name__}")
    if not meta.get("candidates"):
        raise ValueError("json slate missing 'candidates' list")
    return meta


def parse_slate_or_patch(raw_response: str) -> dict:
    """The json block of an answer: a full slate (has `candidates`) or a repair
    patch (`add_candidates` / `replace_candidates` / `set_fields` /
    `add_experience`). Raises ValueError when there is no usable block."""
    json_block = extract_fenced(raw_response, "json")
    meta = json.loads(json_block)
    if not isinstance(meta, dict):
        raise ValueError(f"json block must be an object, got {type(meta).__name__}")
    patch_keys = {"add_candidates", "replace_candidates", "set_fields", "add_experience"}
    if not meta.get("candidates") and not (patch_keys & set(meta)):
        raise ValueError("json block has neither a `candidates` slate nor repair fields")
    # Key drift seen late in long sessions:
    # `workflow` for `name`, `anchor` for `tier_mode_anchors`, no `builds_on` on a `new`.
    entries = list(meta.get("candidates") or []) + list(meta.get("add_candidates") or []) \
        + list((meta.get("replace_candidates") or {}).values())
    for c in entries:
        if not isinstance(c, dict):
            continue
        if not c.get("name") and isinstance(c.get("workflow"), str):
            c["name"] = c.pop("workflow")
        if "tier_mode_anchors" not in c:
            if isinstance(c.get("anchor"), str):
                c["tier_mode_anchors"] = [c.pop("anchor")]
            elif isinstance(c.get("anchors"), list):
                c["tier_mode_anchors"] = c.pop("anchors")
        if "builds_on" not in c and c.get("kind") == "new":
            c["builds_on"] = []
    return meta


def merge_slate_patch(base: dict, patch: dict) -> dict:
    """Apply a repair patch to the previous slate (see parse_slate_or_patch)."""
    meta = json.loads(json.dumps(base))  # deep copy
    cands = [c for c in meta.get("candidates", []) if isinstance(c, dict)]
    by_name = {c.get("name"): c for c in cands}
    for name, fields in (patch.get("set_fields") or {}).items():
        if name in by_name and isinstance(fields, dict):
            by_name[name].update(fields)
    for name, cand in (patch.get("replace_candidates") or {}).items():
        if isinstance(cand, dict):
            cands = [cand if c.get("name") == name else c for c in cands]
            if name not in by_name:
                cands.append(cand)
    for cand in patch.get("add_candidates") or []:
        if isinstance(cand, dict) and cand.get("name") not in {c.get("name") for c in cands}:
            cands.append(cand)
    meta["candidates"] = cands
    exp = [e for e in meta.get("experience", []) or [] if isinstance(e, dict)]
    exp.extend(e for e in (patch.get("add_experience") or []) if isinstance(e, dict))
    meta["experience"] = exp
    for key in ("search_queries", "read_cards"):
        if patch.get(key):
            meta[key] = list(dict.fromkeys((meta.get(key) or []) + list(patch[key])))
    return meta


def describe_patch(patch: dict) -> str:
    parts = []
    for key in ("add_candidates", "replace_candidates", "set_fields", "add_experience"):
        v = patch.get(key)
        if v:
            parts.append(f"{key}={len(v)}")
    return ", ".join(parts) or "(empty patch)"


def build_repair_prompt(meta: dict, violations: list[str], measured_last: list[str]) -> str:
    """Ask only for what the checks found missing, with the strategist's own
    slate echoed back, instead of re-asking for a whole new answer."""
    slate_view = json.dumps({
        "candidates": meta.get("candidates", []),
        "experience": [{"workflow": e.get("workflow")} for e in meta.get("experience", []) or []
                       if isinstance(e, dict)],
    }, indent=1, ensure_ascii=False)
    bullets = "\n".join(f"- {v}" for v in violations)
    return (
        "# Repair your slate (reply with only the repairs)\n\n"
        "Your answer was checked and is incomplete or invalid:\n"
        f"{bullets}\n\n"
        "Do NOT resend the whole answer. Reply with exactly one ```json block containing only the "
        "repairs, in this form (use only the keys you need):\n\n"
        "```json\n{\n"
        '  "add_candidates": [ {"name": "...", "kind": "new|composed|reuse", "target": "accuracy|efficiency|balanced", '
        '"tier_mode_anchors": ["multi"], "why": "...", "builds_on": []} ],\n'
        '  "replace_candidates": { "<existing name>": { ...a full candidate entry... } },\n'
        '  "set_fields": { "<existing name>": { "target": "efficiency" } },\n'
        '  "add_experience": [ {"workflow": "<name>", "observation": "...", "recommendation": "..."} ]\n'
        "}\n```\n\n"
        "Rules: the final slate must have exactly 5 candidates, at least one of each kind and of each "
        "target, every candidate with name/kind/target/tier_mode_anchors/why/builds_on. A candidate you add "
        "or replace with kind `new` or `composed` needs its `# WORKFLOW: <name>` python block in this reply "
        "(fresh name, no structural duplicate of an existing blueprint); a `reuse` needs none. "
        + (f"Blueprints measured last iteration that need an experience entry: "
           f"{', '.join(f'`{n}`' for n in measured_last)}. " if measured_last else "")
        + "Observations and recommendations must be concrete (numbers, failure modes, when to use/avoid).\n\n"
        f"## Your slate as it stands\n\n```json\n{slate_view}\n```\n"
    )


def resolve_workflow(name: str):
    """Resolve a slate name to its callable by importing
    strategist.workflows.<name> and pulling its `execute` attribute.
    Returns (callable, None) on success or (None, error_str) on failure.
    """
    if not _NAME_RE.match(name):
        return None, f"invalid name format (must match `^[a-z][a-z0-9_]*$`)"
    try:
        import importlib
        mod = importlib.import_module(f"strategist.workflows.{name}")
    except ImportError as e:
        return None, f"ImportError: {e}"
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"
    fn = getattr(mod, "execute", None)
    if fn is None or not callable(fn):
        return None, f"module has no callable `execute` attribute"
    return fn, None


def resolve_strategy(name: str, prompts: dict | None = None):
    """A candidate strategy is a library blueprint, optionally paired with
    task-specific prompts (paper Sec. 3, step 2). Without `prompts` the
    blueprint runs with its default prompts; with them, the named prompt
    slots are replaced for this run only and the library file stays as it is."""
    if not prompts:
        return resolve_workflow(name)
    if not _NAME_RE.match(name):
        return None, "invalid name format (must match `^[a-z][a-z0-9_]*$`)"
    return load_with_overrides(name, prompts, STRATEGIST_WORKFLOWS_DIR)


# Below this many hidden rows a single split cannot rank 50 candidates: on
# aqua_rat (18 rows) one question is 5.6pp, and attempts 3-5 picked cheap,
# lucky candidates that then lost on test. Such benchmarks rank by the total
# score over the three development splits instead (reported as a deviation
# from Alg. 1); larger benchmarks keep the paper's hidden-only rule.
SMALL_HIDDEN_N = 100


def selection_rule(hds: dict) -> str:
    return "hidden_accuracy" if int(hds.get("total", 0) or 0) >= SMALL_HIDDEN_N else "total_score_3_splits"


def selection_key(fbs: dict, bls: dict, hds: dict) -> tuple[float, float]:
    """Ranking key of a (candidate, anchor): larger wins. Paper Alg. 1 — highest
    hidden accuracy, ties by lower hidden cost — when the hidden split has at
    least SMALL_HIDDEN_N rows; otherwise the summed score over feedback, blind
    and hidden rows, ties by lower total cost."""
    if selection_rule(hds) == "hidden_accuracy":
        return (float(hds.get("accuracy", 0) or 0), -float(hds.get("total_weighted_cost", 0) or 0))
    score = sum(float(x.get("accuracy", 0) or 0) * int(x.get("total", 0) or 0) for x in (fbs, bls, hds))
    cost = sum(float(x.get("total_weighted_cost", 0) or 0) for x in (fbs, bls, hds))
    return (score, -cost)


def remove_orphan_blueprints(written: set[str], keep: set[str]) -> list[str]:
    """Delete blueprint files written in this iteration that the accepted slate
    neither names nor imports (or all of them when no slate was accepted).
    They were never measured, and every library entry must carry a card."""
    removed: list[str] = []
    imported: set[str] = set()
    for name in keep:
        path = STRATEGIST_WORKFLOWS_DIR / f"{name}.py"
        if path.exists():
            try:
                imported.update(re.findall(r"from strategist\.workflows\.(\w+) import", path.read_text()))
            except OSError:
                pass
    for name in sorted(written - keep - imported):
        path = STRATEGIST_WORKFLOWS_DIR / f"{name}.py"
        try:
            if path.exists():
                path.unlink()
                removed.append(name)
            card = CARDS_DIR / f"{name}.md"
            if card.exists():
                card.unlink()
        except OSError as exc:
            print(f"  [library] could not remove orphan `{name}`: {exc}")
    if removed:
        print(f"  [library] removed {len(removed)} unmeasured blueprint(s) from rejected attempts: "
              f"{', '.join(removed)}")
    return removed


def closing_experience_pass(
    benchmark: str,
    names: list[str],
    results_block: str,
    commander_fn,
    commander: str,
    prompts_dir: Path,
    raw_dir: Path,
    log_path: Path,
    commander_kw: dict | None = None,
) -> int:
    """One more strategist call after the last iteration: experience notes for
    the final slate's blueprints (no new candidates). Returns the number recorded.
    `commander_kw` carries the run's session, when it keeps one."""
    from strategist.experience import record_experience
    from strategist.cards import write_measurement_card
    prompt = (
        f"# Closing experience notes — {benchmark}\n\n"
        "The search on this benchmark is over. The last slate was measured but no further "
        "iteration will write about it. Below are its results. Write one `experience` entry for "
        "each of these blueprints: " + ", ".join(f"`{n}`" for n in names) + ".\n\n"
        "Each entry needs a concrete `observation` (its accuracies and cost per sample, what the "
        "failed traces show, how it compared with the other candidates and the bar) and a concrete "
        "`recommendation` (when to use it, when to avoid it, what to change). Generic one-liners are "
        "useless. Do not propose candidates. Answer with exactly one ```json block:\n\n"
        "```json\n{\n  \"experience\": [\n    {\"workflow\": \"<name>\", \"observation\": \"...\", "
        "\"recommendation\": \"...\"}\n  ]\n}\n```\n\n"
        f"## Results of the last slate\n\n{results_block}\n"
    )
    prompts_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)
    (prompts_dir / "closing_experience.md").write_text(prompt)
    raw = commander_fn(prompt, **(commander_kw or {}))
    (raw_dir / "closing_experience.md").write_text(raw or "")
    meta = json.loads(extract_fenced(raw, "json"))
    kept = 0
    entries = meta.get("experience") or []
    for exp in entries:
        if isinstance(exp, dict) and exp.get("workflow") in names and exp.get("observation") and exp.get("recommendation"):
            record_experience(
                workflow_name=exp["workflow"],
                commander_id=f"{commander}_{benchmark}",
                task_type=benchmark,
                observation=exp["observation"],
                recommendation=exp["recommendation"],
                accuracy_impact=exp.get("accuracy_impact", ""),
            )
            kept += 1
            try:
                write_measurement_card(exp["workflow"])
            except Exception:
                pass
    append_jsonl(log_path, {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"), "benchmark": benchmark,
        "closing_experience": entries, "recorded": kept, "expected": names,
    })
    print(f"  [closing-experience] recorded {kept}/{len(names)} notes for the last slate")
    return kept


def audit_slate(
    candidates_meta: list[dict],
    written_this_iter: set[str],
    ingest_warnings: list[str],
) -> list[str]:
    """Non-fatal audit of the 5-candidate slate composition. Returns a list
    of warning strings (empty = perfect slate).

    Enforces:
      - each candidate has a valid `kind`
      - slate has ≥1 of each kind (`new`, `composed`, `reuse`)
      - name format and banned-prefix check
      - `new` / `composed` names must appear in `written_this_iter` OR
        collide with an existing library file (downgrade-to-reuse path)
      - `reuse` names must NOT appear in `written_this_iter`
      - slate names must resolve via `strategist.workflows.<name>`
      - `builds_on` names must themselves resolve (existing or just written)
    """
    warnings: list[str] = list(ingest_warnings)
    kinds = [c.get("kind") for c in candidates_meta]
    for i, k in enumerate(kinds):
        if k not in VALID_KINDS:
            warnings.append(
                f"candidate {i} ({candidates_meta[i].get('name','?')}): "
                f"missing/invalid `kind` (got {k!r}; expected one of {sorted(VALID_KINDS)})"
            )
    for required in ("new", "composed", "reuse"):
        if kinds.count(required) < 1:
            warnings.append(
                f"slate has 0 `{required}` candidates — aim for ≥1 of each kind"
            )

    for c in candidates_meta:
        name = c.get("name", "")
        kind = c.get("kind")
        if not name:
            warnings.append(f"candidate has empty `name`")
            continue
        if not _NAME_RE.match(name):
            warnings.append(
                f"candidate `{name}` fails name format `^[a-z][a-z0-9_]*$`"
            )
        if any(name.startswith(p) for p in _BANNED_PREFIXES):
            warnings.append(
                f"candidate `{name}` uses banned prefix "
                f"({', '.join(_BANNED_PREFIXES)}) — rename to describe the strategy"
            )

        lib_exists = (STRATEGIST_WORKFLOWS_DIR / f"{name}.py").exists()
        if kind in ("new", "composed"):
            if name not in written_this_iter and not lib_exists:
                warnings.append(
                    f"candidate `{name}` kind={kind} was not written to the library "
                    f"this iteration (missing `# WORKFLOW: {name}` block) and does "
                    f"not already exist — will fail to resolve"
                )
            elif name not in written_this_iter and lib_exists:
                warnings.append(
                    f"candidate `{name}` kind={kind} collides with existing library "
                    f"file — treat as kind=`reuse` next iteration"
                )
        elif kind == "reuse":
            if name in written_this_iter:
                warnings.append(
                    f"candidate `{name}` kind=`reuse` but was also written to the "
                    f"library this iteration — drop the `# WORKFLOW: {name}` block "
                    f"or change kind to `new`/`composed`"
                )
            if not lib_exists:
                warnings.append(
                    f"candidate `{name}` kind=`reuse` but not present in the library — "
                    f"name is unknown"
                )

        builds_on = c.get("builds_on") or []
        if kind == "composed" and not builds_on:
            warnings.append(
                f"candidate `{name}` kind=`composed` has empty `builds_on` — must "
                f"name the library workflows it wraps"
            )
        for b in builds_on:
            if not (STRATEGIST_WORKFLOWS_DIR / f"{b}.py").exists() and b not in written_this_iter:
                warnings.append(
                    f"candidate `{name}` builds_on `{b}` which doesn't resolve "
                    f"(not in library, not written this iteration)"
                )

        # Paper A.4: non-fatal AST audit of the candidate's code — `builds_on`
        # must match the actual blueprint imports, and a dict-signature
        # workflow must be paired with a "multi" anchor (the top silent
        # failure mode: `llms[...]` subscripting on a single client).
        wf_path = STRATEGIST_WORKFLOWS_DIR / f"{name}.py"
        tree = None
        if wf_path.exists():
            try:
                tree = ast.parse(wf_path.read_text())
            except (SyntaxError, OSError):
                tree = None
        if tree is not None:
            if name in written_this_iter:
                imported: set[str] = set()
                for node in ast.walk(tree):
                    if (isinstance(node, ast.ImportFrom) and node.module
                            and node.module.startswith("strategist.workflows.")):
                        imported.add(node.module.split(".")[-1])
                    elif isinstance(node, ast.Import):
                        for alias in node.names:
                            if alias.name.startswith("strategist.workflows."):
                                imported.add(alias.name.split(".")[-1])
                imported.discard("_qa_common")
                for b in set(builds_on) - imported - {name}:
                    warnings.append(
                        f"candidate `{name}` declares builds_on `{b}` but its code "
                        f"never imports strategist.workflows.{b}"
                    )
                for m in imported - set(builds_on):
                    warnings.append(
                        f"candidate `{name}` imports strategist.workflows.{m} but "
                        f"omits it from builds_on"
                    )
            first_param = None
            for node in ast.walk(tree):
                if (isinstance(node, (ast.AsyncFunctionDef, ast.FunctionDef))
                        and node.name == "execute"):
                    if node.args.args:
                        first_param = node.args.args[0].arg
                    break
            anchors = c.get("tier_mode_anchors") or []
            if first_param == "llms" and anchors and "multi" not in anchors:
                warnings.append(
                    f"candidate `{name}` has an `llms` dict signature but no "
                    f"\"multi\" anchor — single anchors pass one client and "
                    f"`llms[tier][mode]` subscripting will crash every sample"
                )
            elif first_param == "llm" and anchors == ["multi"]:
                warnings.append(
                    f"candidate `{name}` has a single-client `llm` signature but "
                    f"only a \"multi\" anchor — it will silently run every call "
                    f"at balanced/nothink; declare a specific anchor if that is "
                    f"not the intent"
                )

    return warnings


REQUIRED_CANDIDATE_FIELDS = ("name", "kind", "tier_mode_anchors", "why", "builds_on", "target")


def slate_violations(
    candidates_meta: list[dict],
    ingest_warnings: list[str],
    expected_k: int = 5,
    require_kinds: bool = True,
) -> list[str]:
    """FATAL slate checks (paper A.4): a slate that omits required JSON
    fields, misses a kind, or structurally duplicates a recent candidate is
    rejected and the strategist re-prompted. Distinct from `audit_slate`,
    whose warnings are non-fatal and fed into the next iteration."""
    violations: list[str] = []
    if len(candidates_meta) != expected_k:
        violations.append(
            f"slate has {len(candidates_meta)} candidates; exactly {expected_k} required"
        )
    for i, c in enumerate(candidates_meta):
        if not isinstance(c, dict):
            violations.append(f"candidate {i} is not a JSON object")
            continue
        missing = [f for f in REQUIRED_CANDIDATE_FIELDS if f not in c]
        if missing:
            violations.append(
                f"candidate {i} ({c.get('name', '?')}) missing required field(s): "
                f"{', '.join(missing)}"
            )
        if c.get("kind") not in VALID_KINDS:
            violations.append(
                f"candidate {i} ({c.get('name', '?')}) has invalid kind "
                f"{c.get('kind')!r}; must be one of {sorted(VALID_KINDS)}"
            )
        if c.get("target") not in VALID_TARGETS:
            violations.append(
                f"candidate {i} ({c.get('name', '?')}) has invalid target "
                f"{c.get('target')!r}; must be one of {sorted(VALID_TARGETS)}"
            )
        if c.get("prompts"):
            wf_path = STRATEGIST_WORKFLOWS_DIR / f"{c.get('name', '')}.py"
            try:
                problems = override_problems(wf_path.read_text(), c["prompts"])
            except (OSError, SyntaxError) as exc:
                problems = [f"blueprint could not be read ({exc})"]
            for problem in problems:
                violations.append(f"candidate {i} ({c.get('name', '?')}) `prompts`: {problem}")
    # With the library disabled (ablations) there is nothing to reuse or
    # compose in the first iteration, so the one-of-each-kind rule is waived.
    kinds = [c.get("kind") for c in candidates_meta if isinstance(c, dict)]
    for required in sorted(VALID_KINDS) if require_kinds else ():
        if kinds.count(required) < 1:
            violations.append(f"slate has no `{required}` candidate; at least one of each kind is required")
    targets = [c.get("target") for c in candidates_meta if isinstance(c, dict)]
    for required in sorted(VALID_TARGETS):
        if targets.count(required) < 1:
            violations.append(f"slate has no `target: {required}` candidate; every slate needs an "
                              f"accuracy, an efficiency and a balanced candidate")
    for w in ingest_warnings:
        if "structural duplicate" in w:
            violations.append(w)
    return violations


def workflow_signature_kind(fn: Callable) -> str:
    """Return 'llm' (single-client) or 'llms' (full dict) based on first
    positional parameter name."""
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return "llm"
    params = list(sig.parameters.values())
    if not params:
        return "llm"
    first = params[0].name
    return "llms" if first == "llms" else "llm"


# ---------------------------------------------------------------------------
# Per-sample eval
# ---------------------------------------------------------------------------


async def eval_one(
    workflow_fn: Callable,
    sample: dict,
    llms: dict[str, dict[str, types.SimpleNamespace]],
    anchor_client: types.SimpleNamespace | None,
    expected_field: str,
    per_sample_timeout: int,
) -> dict:
    task = str(sample.get("task", ""))
    expected = str(sample.get(expected_field, ""))
    difficulty = str(sample.get("difficulty", "unknown"))
    benchmark = str(sample.get("benchmark", ""))

    cost_records: list[dict] = []
    _cost_var.set(cost_records)

    kind = workflow_signature_kind(workflow_fn)
    start = time.perf_counter()
    try:
        if kind == "llms":
            coro = workflow_fn(llms, task)
        else:
            # Single-client signature; pass the anchor client (or fast/nothink
            # as a sane default if no anchor was specified).
            coro = workflow_fn(anchor_client, task)
        # No artificial per-sample timeout. SLURM walltime is the only
        # outer bound. Earlier runs lost ~33,500 candidates to a 900s cap
        # that killed legitimate complex/thinking multi-call pipelines
        # mid-execution. Trust the workflow to finish or error naturally.
        if per_sample_timeout and per_sample_timeout > 0:
            output = await asyncio.wait_for(coro, timeout=per_sample_timeout)
        else:
            output = await coro
        error = None
    except asyncio.TimeoutError:
        output = ""
        error = None  # timeout = legitimate fail (model too slow), not infra error
    except Exception as exc:
        output = ""
        error = f"{type(exc).__name__}: {exc}"
    elapsed = time.perf_counter() - start

    passed = False
    score = 0.0
    note = ""
    if not error and expected:
        if sample.get("test_code"):
            passed, _src, note = _evaluate_humaneval_completion(task, sample, output)
            score = 1.0 if passed else 0.0
        else:
            from strategist.scoring import grade
            g = grade(benchmark, task, expected, str(output), refs=sample.get("answers"))
            passed, score, note = g["passed"], g["score"], g["note"]

    weighted_cost = sum(
        (r["in_tok"] + r["out_tok"]) * TIER_WEIGHTS.get(r["tier"], 1)
        for r in cost_records
    )

    return {
        "task": task,
        "expected": expected,
        "output": str(output) if output else "",
        "difficulty": difficulty,
        "passed": bool(passed),
        "score": float(score),
        "error": error,
        "evaluation_note": note,
        "latency_seconds": round(elapsed, 2),
        "n_calls": len(cost_records),
        "cost_records": cost_records,
        "weighted_cost": weighted_cost,
    }


def _save_generations(record_dir: Path, samples: list[dict], results: list[dict]) -> None:
    """Write every executor call of an evaluation in full (generations.jsonl:
    request and response per call, agent transcripts included) and one line per
    sample (samples.jsonl), then drop the heavy fields from the in-memory
    records so logs and result files keep their earlier size."""
    record_dir.mkdir(parents=True, exist_ok=True)
    with (record_dir / "generations.jsonl").open("a", encoding="utf-8") as g, \
            (record_dir / "samples.jsonl").open("a", encoding="utf-8") as f:
        for idx, (sample, r) in enumerate(zip(samples, results)):
            for call, rec in enumerate(r["cost_records"]):
                g.write(json.dumps({"idx": idx, "call": call, **rec}, ensure_ascii=False) + "\n")
            light = {k: v for k, v in r.items() if k != "cost_records"}
            light["idx"] = idx
            light["in_tok"] = sum(int(rec.get("in_tok", 0) or 0) for rec in r["cost_records"])
            light["out_tok"] = sum(int(rec.get("out_tok", 0) or 0) for rec in r["cost_records"])
            if sample.get("answers") is not None:
                light["answers"] = sample["answers"]
            f.write(json.dumps(light, ensure_ascii=False) + "\n")


def _strip_heavy(results: list[dict]) -> None:
    for r in results:
        r["cost_records"] = [{k: v for k, v in rec.items() if k not in _HEAVY_RECORD_KEYS}
                             for rec in r["cost_records"]]


async def eval_candidate(
    workflow_fn: Callable,
    samples: list[dict],
    llms: dict[str, dict[str, types.SimpleNamespace]],
    anchor: str,
    expected_field: str,
    concurrency: int,
    per_sample_timeout: int,
    record_dir: Path | None = None,
) -> dict:
    """Run one workflow against one anchor on a sample set. When anchor is
    "multi" the workflow gets the full llms dict; otherwise every entry is
    redirected to one (tier, mode) client. With `record_dir`, every executor
    call is saved there in full (see _save_generations)."""
    if anchor == "multi":
        effective = llms
        anchor_client = llms["balanced"]["nothink"]
    else:
        effective = redirect_all_to(llms, anchor)
        tier, mode = anchor.split("/")
        anchor_client = llms[tier][mode]

    semaphore = asyncio.Semaphore(max(1, concurrency))

    async def run_one(sample: dict) -> dict:
        # Each asyncio task inherits its own copy of the contextvar; eval_one
        # sets a fresh list per call.
        async with semaphore:
            return await eval_one(
                workflow_fn=workflow_fn,
                sample=sample,
                llms=effective,
                anchor_client=anchor_client,
                expected_field=expected_field,
                per_sample_timeout=per_sample_timeout,
            )

    results = await asyncio.gather(*(run_one(s) for s in samples))
    if record_dir is not None:
        try:
            _save_generations(record_dir, samples, results)
        except Exception as exc:  # a failed record must not fail the evaluation
            print(f"  [warn] could not save generations to {record_dir}: {type(exc).__name__}: {exc}", flush=True)
    _strip_heavy(results)
    total = len(results)
    passed = sum(1 for r in results if r["passed"])
    by_tier_mode: dict[str, dict[str, int]] = {}
    for r in results:
        for rec in r["cost_records"]:
            key = f"{rec['tier']}/{rec['mode']}"
            bm = by_tier_mode.setdefault(key, {"calls": 0, "in_tok": 0, "out_tok": 0})
            bm["calls"] += 1
            bm["in_tok"] += rec["in_tok"]
            bm["out_tok"] += rec["out_tok"]
    total_weighted = sum(r["weighted_cost"] for r in results)

    # Aggregate error types: first 80 chars of each error string, counted.
    # Useful for diagnosing 0%-accuracy candidates without opening sample logs.
    from collections import Counter
    err_counter: Counter = Counter()
    for r in results:
        if r.get("error"):
            err_counter[str(r["error"])[:80]] += 1
    top_errors = [{"msg": m, "count": c} for m, c in err_counter.most_common(3)]

    return {
        "summary": {
            "total": total,
            "passed": passed,
            # mean per-sample score: 1/0 everywhere, token F1 on SQuAD v2 (paper A.2)
            "accuracy": sum(r.get("score", 1.0 if r["passed"] else 0.0) for r in results) / total if total else 0,
            "errors": sum(1 for r in results if r["error"]),
            "avg_latency_seconds": sum(r["latency_seconds"] for r in results) / total if total else 0,
            "total_weighted_cost": total_weighted,
            "by_tier_mode": by_tier_mode,
            "top_errors": top_errors,
        },
        "samples": results,
    }


# ---------------------------------------------------------------------------
# Feedback rendering (what the commander sees next iteration)
# ---------------------------------------------------------------------------


def _fmt_by_tier_mode(by_tier_mode: dict) -> str:
    if not by_tier_mode:
        return "  (no calls)"
    lines = []
    for key in sorted(by_tier_mode.keys()):
        d = by_tier_mode[key]
        lines.append(
            f"  {key:20s} calls={d['calls']:3d} in_tok={d['in_tok']:7d} out_tok={d['out_tok']:7d}"
        )
    return "\n".join(lines)


def _fmt_failed_traces(samples: list[dict], limit: int = 5) -> str:
    failed = [s for s in samples if not s["passed"]][:limit]
    if not failed:
        return "  (no failures)"
    parts = []
    for i, s in enumerate(failed, 1):
        head = f"  #{i} diff={s['difficulty']} err={s['error'] or 'wrong_answer'}"
        body = (
            f"    Q: {s['task'][:200]}\n"
            f"    Expected: {s['expected'][:100]} | Got: {s['output'][:120]}\n"
        )
        cost_lines = []
        for rec in s["cost_records"][:6]:
            line = f"      [{rec['tier']}/{rec['mode']}] in={rec['in_tok']} out={rec['out_tok']} t={rec['latency']}s"
            tail = " ".join(str(rec.get("out_tail") or "").split())
            if tail and i <= 3:
                # what this step produced (end of its output), first 3 failures only
                line += f"\n        -> ...{tail[-TRACE_SHOWN_CHARS:]}"
            cost_lines.append(line)
        cost_block = "\n".join(cost_lines) if cost_lines else "    (no LLM calls recorded)"
        parts.append(head + "\n" + body + cost_block)
    return "\n\n".join(parts)


_STRUCTURE_PATTERNS = [
    ("tool use (agent)", r"await agent\("),
    ("voting / sampling", r"sample_n_answers|majority_vote|Counter\("),
    ("decomposition", r"decompos|sub[-_ ]?question|subtask|sub-problem"),
    ("multi-agent roles / debate", r"\brole|debate|persona"),
    ("routing / escalation", r"escalat|\broute|if .*(disagree|confiden|uncertain|agree)"),
    ("draft → check → normalize", r"verif|audit|check|critique|falsif"),
]


def render_run_structure(log_path: Path) -> str:
    """One line on the shapes of the blueprints this run has written so far —
    the mirror the strategist needs to see when its 'new' candidates are all
    the same pipeline with new wording."""
    names: list[str] = []
    if log_path.exists():
        for line in log_path.open():
            try:
                names.extend(json.loads(line).get("library_written") or [])
            except Exception:
                continue
    names = list(dict.fromkeys(names))
    if not names:
        return "(none yet — this is the first slate)"
    counts = {label: 0 for label, _ in _STRUCTURE_PATTERNS}
    plain = 0
    for name in names:
        try:
            code = (STRATEGIST_WORKFLOWS_DIR / f"{name}.py").read_text()
        except OSError:
            continue
        hit = False
        for label, rx in _STRUCTURE_PATTERNS:
            if re.search(rx, code, re.I):
                counts[label] += 1
                hit = True
        if not hit:
            plain += 1
    parts = [f"{v} {k}" for k, v in counts.items() if v] + ([f"{plain} single-call prompt"] if plain else [])
    return (f"Your {len(names)} blueprints so far: " + ", ".join(parts)
            + ". The innovative candidate must be structurally unlike all of these.")


def render_iteration_history(log_path: Path, benchmark: str, limit: int = 10) -> str:
    """Prior-iteration log for the strategist prompt (paper A.5): every earlier
    iteration's candidates with their training (feedback) and validation
    (blind) accuracy and cost. Hidden-set and test numbers are never included.

    The loop used to call the v1 reader here, which looks for an `eval_summary`
    field that log entries do not have and so printed `acc=0.0% (0/0)` for
    every past iteration."""
    if not log_path or not Path(log_path).exists():
        return ""
    entries: list[dict] = []
    for raw in Path(log_path).read_text().splitlines():
        try:
            entry = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if entry.get("benchmark") != benchmark or entry.get("final_test"):
            continue
        if "iteration" in entry:
            entries.append(entry)
    blocks: list[str] = []
    for entry in entries[-limit:]:
        head = f"iteration {entry['iteration']}:"
        if entry.get("error"):
            blocks.append(f"{head} no slate evaluated ({str(entry['error'])[:120]})")
            continue
        kinds = {c.get("name"): c.get("kind") for c in entry.get("candidates_meta") or [] if isinstance(c, dict)}
        lines = [head]
        for c in entry.get("candidates") or []:
            fs, bs = c.get("feedback_summary") or {}, c.get("blind_summary") or {}
            n = max(int(fs.get("total", 0) or 0), 1)
            cps = float(fs.get("total_weighted_cost", 0) or 0) / n
            blind = (f"blind vs the reference: {c.get('blind_direction') or 'n/a'}"
                     if paired_pick.blind_direction_only()
                     else f"blind {float(bs.get('accuracy', 0) or 0):.1%}")
            lines.append(
                f"  - `{c.get('candidate')}` ({kinds.get(c.get('candidate'), '?')}) @ `{c.get('anchor')}`: "
                f"feedback {float(fs.get('accuracy', 0) or 0):.1%}, {blind}, "
                f"cost/sample {cps:.0f}, errors {int(fs.get('errors', 0) or 0)}"
            )
        if len(lines) == 1:
            lines.append("  (no candidate was evaluated)")
        blocks.append("\n".join(lines))
    return "\n".join(blocks)


def blind_shown_line(direction: str | None) -> str:
    """The blind line of a direction-only run (--blind-feedback direction)."""
    return f"- blind:    {direction or 'direction not available'} (vs the reference; direction only)"


def render_candidate_results(
    candidate_name: str,
    anchor: str,
    feedback_eval: dict,
    blind_eval: dict,
    why: str,
    blind_direction: str | None = None,
) -> str:
    """`blind_direction` replaces the blind numbers in a direction-only run."""
    fs = feedback_eval["summary"]
    bs = blind_eval["summary"]
    f_total = max(int(fs.get("total", 0) or 0), 1)
    b_total = max(int(bs.get("total", 0) or 0), 1)
    f_cps = float(fs.get("total_weighted_cost", 0) or 0) / f_total
    b_cps = float(bs.get("total_weighted_cost", 0) or 0) / b_total
    if paired_pick.blind_direction_only():
        blind = blind_shown_line(blind_direction) + "\n"
    else:
        blind = (f"- blind:    acc={bs['accuracy']:.1%} ({bs['passed']}/{bs['total']}), "
                 f"errors={bs['errors']}, **cost/sample = {b_cps:.0f}** (total={bs['total_weighted_cost']:.2e})\n")
    return (
        f"### `{candidate_name}` @ `{anchor}`\n"
        f"why: {why or '—'}\n\n"
        f"- feedback: acc={fs['accuracy']:.1%} ({fs['passed']}/{fs['total']}), "
        f"errors={fs['errors']}, **cost/sample = {f_cps:.0f}** (total={fs['total_weighted_cost']:.2e})\n"
        f"{blind}"
        f"per-tier call breakdown (feedback set):\n"
        f"{_fmt_by_tier_mode(fs['by_tier_mode'])}\n\n"
        f"failed traces (feedback set, up to 5):\n"
        f"{_fmt_failed_traces(feedback_eval['samples'])}\n"
    )


# ---------------------------------------------------------------------------
# Card + search injection
# ---------------------------------------------------------------------------


CARDS_DIR = STRATEGIST_DIR / "cards"
LIBRARY_META_DIR = STRATEGIST_DIR / "library_meta"


def _designed_for(name: str) -> str:
    """' — designed for <benchmark> (<kind>)' for blueprints a search added."""
    try:
        from strategist.retrieval import library_meta
        side = library_meta(name)
    except Exception:
        side = {}
    if not side.get("designed_for"):
        return ""
    kind = f" ({side['task_kind']})" if side.get("task_kind") else ""
    return f" — designed for `{side['designed_for']}`{kind}"


def load_requested_cards(names: list[str]) -> str:
    if not names:
        return "(none requested)"
    parts = []
    for name in names:
        path = CARDS_DIR / f"{name}.md"
        if not path.exists():
            parts.append(f"### {name}\n(card not found)")
            continue
        parts.append(f"### {name}{_designed_for(name)}\n\n{path.read_text()}")
    return "\n\n---\n\n".join(parts)


def run_library_retrieval(
    queries: list[str],
    fallback_task: str,
) -> tuple[str, str]:
    """Library retrieval per paper A.4: two textual queries against the
    library — a relevance query summarising the task, and an inspiration
    query phrased to deliberately drift from it. Each query is scored by the
    hybrid function s = 0.4*tfidf + 0.6*embedding over every blueprint's
    index entry; each returns its top-5, with the inspiration slate
    deduplicated against the relevance hits.

    Returns (relevant_block, inspiration_block): the 5 relevance hits are
    rendered with their measurement cards in full; the 5 adjacent
    inspirations with only their one-line description.
    """
    if library_size() == 0:
        return "(the library is empty: nothing to retrieve yet)", "(the library is empty)"
    try:
        from strategist.retrieval import retrieve
    except Exception as e:
        return f"(search unavailable: {e})", "(search unavailable)"

    relevance_query = (queries[0] if queries else "") or fallback_task
    inspiration_query = " ".join(queries[1:]).strip() or (
        "diverse reasoning patterns from other domains: multi-path voting, "
        "verification, decomposition, tool use, long-context evidence chaining"
    )
    try:
        relevant, inspiration = retrieve(relevance_query, inspiration_query, k=5)
    except Exception as e:
        return f"(retrieval failed: {e})", "(retrieval failed)"

    index = {}
    try:
        index = json.loads((STRATEGIST_DIR / "index.json").read_text()).get("workflows", {})
    except Exception:
        pass

    rel_parts: list[str] = []
    for name, score in relevant:
        card_path = CARDS_DIR / f"{name}.md"
        head = f"### `{name}` (relevance {score:.2f}){_designed_for(name)}"
        if card_path.exists():
            block = f"{head}\n\n{card_path.read_text()}"
        else:
            desc = index.get(name, {}).get("description", "(no card yet)")
            block = f"{head}\n\n{desc}"
        # The blueprint's default prompts, so a `reuse` candidate can pair it
        # with task-specific ones (candidate field `prompts`).
        slots = describe_slots(name, workflows_dir=STRATEGIST_WORKFLOWS_DIR)
        rel_parts.append(block + (f"\n\n{slots}" if slots else ""))
    insp_parts: list[str] = []
    for name, score in inspiration:
        desc = index.get(name, {}).get("description", "")[:120]
        insp_parts.append(f"- `{name}`{_designed_for(name)}: {desc}")

    return (
        "\n\n---\n\n".join(rel_parts) or "(no relevant blueprints found)",
        "\n".join(insp_parts) or "(no adjacent inspiration found)",
    )


# ---------------------------------------------------------------------------
# Prompt building
# ---------------------------------------------------------------------------


# 10 distinct personas, one per slot. run_seed 0..9 maps directly.
# Slot 0 is the no-framing control; slot 1 is the user-specified data-aware
# router (segment classification + inter-LLM communication example). Slots
# 2-9 are orthogonal strategic biases. Run ids ≥ 10 wrap modulo.
PERSONA_SLOTS = [
    "default",              # 0 — control, shared system prompt only
    "data_aware",           # 1 — segment + route on data structure
    "verification_heavy",   # 2 — every candidate has a verify step
    "failure_mode_driven",  # 3 — target specific failure patterns from prior iters
    "contrarian",           # 4 — invert the reference block's dominant property
    "decomposition_first",  # 5 — explicit sub-question decomposition in every candidate
    "ensemble_vote",        # 6 — N-sample voting with different aggregation rules
    "tool_use_bias",        # 7 — prefer agent() with Python execution
    "multi_only",           # 8 — every candidate uses ≥2 llms[tier][mode] clients
    "tier_ladder",          # 9 — slate spans ≥3 tiers (fast/balanced/complex)
]


def _load_persona(name: str) -> str:
    """Load persona fragment by name. Unknown persona → empty string (= default)."""
    from pathlib import Path
    path = Path(__file__).parent / "prompts" / "personas" / f"{name}.md"
    if not path.exists():
        return ""
    return path.read_text().strip()


def resolve_persona(run_seed: int | None, explicit: str | None) -> str:
    """Pick the persona name for this run.

    The paper's strategist uses ONE fixed prompt template for every run
    (A.5); persona framings are an ablation-only mechanism. They therefore
    activate only via an explicit --prompt-variant — never from run_seed."""
    if explicit:
        return explicit
    return "default"


CONTINUITY_STATELESS = """**Your continuity across iterations lives in the prompt, not in a
   session.** Each iteration you receive the prior-iteration log: your
   previous candidates, their measured accuracy and cost, failed traces,
   and any audit warnings. Build on that record instead of re-analyzing
   the benchmark from scratch. The `experience` JSON block is your
   durable writeback — it is stored in the library and read by
   strategists on other benchmarks, so make every entry concrete.

   Practical consequence: at iter 2+, spend the budget on "given the
   measurement results in the log, what's the next experiment?"
"""

CONTINUITY_SESSION = """**You keep this conversation for the whole run.** After this first
   prompt, each iteration sends only what is new: your last candidates'
   measurements and failed traces, the updated comparison with the
   reference, this round's cards, and a one-line log of every candidate
   so far. The rules, the benchmark and the baseline table stay as given
   here. Build on your own earlier reasoning and results instead of
   re-analyzing the benchmark from scratch: at iter 2+, ask "given what I
   have measured so far, what is the next experiment?" The `experience`
   JSON block is your durable writeback — it is stored in the library and
   read by strategists on other benchmarks, so make every entry concrete.
"""


def blind_shown_text() -> str:
    """What the strategist is told it sees of the blind split ({blind_shown})."""
    if paired_pick.blind_direction_only():
        return ("Accuracy on the feedback split; on the blind split only the direction "
                "against the reference on the same questions: accuracy higher / about the "
                "same / lower, cost lower / about the same / higher (within 2 points or 2% "
                "is about the same).")
    return "Accuracy on the feedback and blind splits."


PROMPTS_DIR = Path(__file__).parent / "prompts"


def load_prompt(name: str) -> str:
    """Load a markdown prompt template from the prompts/ folder."""
    path = PROMPTS_DIR / f"{name}.md"
    return path.read_text(encoding="utf-8")


def build_system_prompt(
    workflow_count: int,
    search_results: str,
    requested_cards: str,
    persona: str = "default",
    resume_session: bool = False,
) -> str:
    """`resume_session` picks the continuity rule: one conversation for the
    run, or a fresh session every iteration (the paper's strategist)."""
    tpl = load_prompt("system")
    rendered = (
        tpl.replace("{workflow_count}", str(workflow_count))
        .replace("{search_results}", search_results)
        .replace("{requested_cards}", requested_cards)
        .replace("{continuity_rule}", CONTINUITY_SESSION if resume_session else CONTINUITY_STATELESS)
        .replace("{blind_shown}", blind_shown_text())
    )
    # Inject persona preamble above the system prompt. Personas are short
    # strategic framings (diversity, exploit, tier_ladder, multi_only) that
    # steer the slate without changing any of the hard rules below.
    persona_text = _load_persona(persona)
    if persona_text:
        rendered = persona_text + "\n\n---\n\n" + rendered
    return rendered


_REFERENCE_BLOCK_RE = re.compile(
    r"<!-- reference:begin -->\n.*?<!-- reference:end -->\n?", re.S
)


def _apply_reference_blocks(tpl: str, enabled: bool) -> str:
    """The marked template regions describe the reference strategy (the bar to
    beat); they are rendered only for a run with --reference."""
    if enabled:
        return tpl.replace("<!-- reference:begin -->\n", "").replace(
            "<!-- reference:end -->\n", ""
        )
    return _REFERENCE_BLOCK_RE.sub("", tpl)


def build_create_prompt(
    benchmark_name: str,
    benchmark_description: str,
    sample_questions: list[dict],
    workflow_count: int,
    workflow_summary: str,
    search_results: str,
    requested_cards: str,
    performance_history: str,
    bar_to_beat: str = "",
    show_reference: bool = False,
    run_structure: str = "",
) -> str:
    tpl = _apply_reference_blocks(load_prompt("create"), show_reference)
    samples_text = ""
    for i, s in enumerate(sample_questions[:5], 1):
        samples_text += (
            f"\n--- Example {i} ---\n"
            f"Question: {s.get('task', '')}\n"
            f"Answer: {s.get('answer', '?')}\n"
        )
    return (
        tpl.replace("{benchmark_name}", benchmark_name)
        .replace("{benchmark_description}", benchmark_description)
        .replace("{bar_to_beat}", bar_to_beat or "(no bar computed)")
        .replace("{sample_questions}", samples_text)
        .replace("{workflow_count}", str(workflow_count))
        .replace("{workflow_summary}", workflow_summary)
        .replace("{search_results}", search_results)
        .replace("{requested_cards}", requested_cards)
        .replace("{performance_history}", performance_history or "(first iteration)")
        .replace("{run_structure}", run_structure or "(none yet)")
    )


def _best_from_run_log(log_path: Path) -> dict | None:
    """Return the best candidate this run has produced, scored by BLIND
    accuracy (not hidden). The commander's visible progress is blind;
    using hidden here would show a number the commander can't correlate
    with its own measurements and would leak selection signal. Ties on
    acc broken by lower cost/sample."""
    if not log_path.exists():
        return None
    best = None
    best_acc = -1.0
    for line in log_path.open():
        try:
            j = json.loads(line)
        except Exception:
            continue
        for c in j.get("candidates") or []:
            bs = c.get("blind_summary") or {}
            total_cost = float(bs.get("total_weighted_cost", 0) or 0)
            if total_cost <= 0:
                continue
            n = max(int(bs.get("total", 0) or 0), 1)
            acc = float(bs.get("accuracy", 0) or 0)
            cps = total_cost / n
            if acc > best_acc or (
                acc == best_acc and best is not None and cps < best["cost_per_sample"]
            ):
                best_acc = acc
                best = {
                    "iteration": j.get("iteration"),
                    "candidate": c.get("candidate"),
                    "anchor": c.get("anchor"),
                    "blind_acc": acc,
                    "cost_per_sample": cps,
                }
    return best


def build_revise_prompt(
    iteration: int,
    benchmark_name: str,
    benchmark_description: str,
    candidate_results: str,
    workflow_count: int,
    workflow_summary: str,
    search_results: str,
    requested_cards: str,
    performance_history: str,
    bar_to_beat: str = "",
    show_reference: bool = False,
    run_structure: str = "",
) -> str:
    tpl = _apply_reference_blocks(load_prompt("revise"), show_reference)
    return (
        tpl.replace("{iteration}", str(iteration))
        .replace("{benchmark_name}", benchmark_name)
        .replace("{benchmark_description}", benchmark_description)
        .replace("{bar_to_beat}", bar_to_beat or "(no bar computed)")
        .replace("{candidate_results}", candidate_results)
        .replace("{workflow_count}", str(workflow_count))
        .replace("{workflow_summary}", workflow_summary)
        .replace("{search_results}", search_results)
        .replace("{requested_cards}", requested_cards)
        .replace("{performance_history}", performance_history or "(none)")
        .replace("{run_structure}", run_structure or "(none yet)")
        .replace("{blind_shown}", blind_shown_text())
    )


_CARD_SPLIT = re.compile(r"\n\n---\n\n(?=### )")


def card_for_prompt(card: str, benchmark: str, max_notes: int = 3, max_measured: int = 4) -> str:
    """A library card as the strategist sees it while working on `benchmark`
    (the card file keeps everything): the baseline-table rows of other
    benchmarks, the card-type banner and run ids are dropped, and only the
    latest `max_notes` notes and `max_measured` measurements are kept."""
    body, tail = card, ""
    k = card.find("\nPrompt slots (")
    if k >= 0:
        body, tail = card[:k], card[k:]
    sections = re.split(r"\n(?=## )", body)
    out = []
    for sec in sections:
        head, _, rest = sec.partition("\n")
        lines = rest.split("\n")
        if not sec.startswith("## "):
            sec = "\n".join(l for l in sec.split("\n") if not l.startswith("> "))
            sec = re.sub(r"\n{3,}", "\n\n", sec)
        elif head.startswith("## Measured behavior (baseline table"):
            keep = [l for l in lines if not l.startswith("| ") or l.startswith(("| benchmark |", "| **mean"))
                    or l.startswith(f"| {benchmark} |")]
            sec = head + "\n" + "\n".join(keep)
        elif head == "## Measured behavior":
            items = [l for l in lines if l.startswith("- ")]
            if paired_pick.blind_direction_only():   # val = the blind split: not shown as numbers
                items = [re.sub(r" val acc=\S+ \([^)]*\)", "", l) for l in items]
            other = [l for l in lines if l.strip() and not l.startswith("- ")]
            mine = [l for l in items if l.startswith(f"- {benchmark} @")]
            shown = (mine + [l for l in items if l not in mine])[:max_measured]
            more = len(items) - len(shown)
            sec = head + "\n" + "\n".join(other + shown + ([f"(+{more} more measurements not shown)"] if more else []))
            sec += "\n"
        elif head.startswith(("## Accumulated experience", "## Notes from prior use")):
            notes, cur, rest_lines = [], None, []
            for l in lines:
                if l.startswith("- ["):
                    cur = [l]; notes.append(cur)
                elif cur is not None and l.startswith("  "):
                    cur.append(l)
                elif l.strip():
                    rest_lines.append(l)
            if notes:
                dropped = len(notes) - max_notes
                kept = notes[-max_notes:]
                sec = head + "\n" + ("(latest notes only)\n" if dropped > 0 else "") + \
                    "\n".join("\n".join(n) for n in kept) + "\n"
        elif head == "## Designed for":
            sec = re.sub(r" of run \S+", "", sec)
        out.append(sec)
    return "\n".join(out) + tail


def cards_for_prompt(block: str, benchmark: str) -> str:
    """card_for_prompt over a rendered card block (relevance hits or requested cards)."""
    if not block or block.startswith("("):
        return block
    parts = []
    for part in _CARD_SPLIT.split(block):
        head, sep, card = part.partition("\n")
        parts.append(head + sep + card_for_prompt(card, benchmark) if part.startswith("### ") else part)
    return "\n\n---\n\n".join(parts)


def card_bodies(block: str) -> dict[str, str]:
    """name -> text of every card in a rendered card block (relevance hits or
    requested cards), without the `### name ...` header line, whose relevance
    score changes from round to round."""
    out = {}
    for part in _CARD_SPLIT.split(block or ""):
        m = re.match(r"### `?([A-Za-z0-9_]+)`?", part)
        if m:
            out[m.group(1)] = part.split("\n", 1)[1] if "\n" in part else ""
    return out


def dedupe_cards(block: str, seen: dict[str, str]) -> str:
    """In a resumed session, a card the strategist already saw is not sent again:
    unchanged, it becomes its header plus a pointer; changed (cards gain notes
    every round), its header plus only the lines that are new, unless most of
    it is new."""
    if not block or not seen:
        return block
    parts = []
    for part in _CARD_SPLIT.split(block):
        m = re.match(r"### `?([A-Za-z0-9_]+)`?", part)
        old = seen.get(m.group(1)) if m else None
        if old is not None:
            head, _, body = part.partition("\n")
            if body == old:
                part = head + "\n(card unchanged since you saw it earlier in this conversation)"
            else:
                known = set(old.splitlines())
                lines = [l for l in body.splitlines() if l.strip()]
                fresh = [l for l in lines if l not in known]
                if len(fresh) <= len(lines) / 2:
                    part = (head + "\n(card updated since you saw it earlier in this conversation; "
                            "only its new lines:)\n" + "\n".join(fresh))
        parts.append(part)
    return "\n\n---\n\n".join(parts)


def restore_round_state(run_log: Path, prompts_dir: Path) -> dict:
    """What the next iteration needs from the last logged one, for a run that
    resumes (after a crash, or extended past its last iteration): the
    strategist's card requests and search queries, the blueprints that still
    owe an experience note, and the results block it is shown. A finished run's
    results block is the one its closing call carried (whose notes are then
    already written); otherwise a summary is rebuilt from the log."""
    last = None
    for line in run_log.open():
        try:
            e = json.loads(line)
        except ValueError:
            continue
        if e.get("iteration") and (e.get("candidates") or e.get("error")):
            last = e
    if last is None:
        return {}
    cands = last.get("candidates") or []
    names = {c.get("candidate") for c in cands if c.get("candidate")}
    out = {"search_queries": last.get("search_queries") or [], "read_cards": last.get("read_cards") or [],
           "names": names, "results_block": None}
    closing = prompts_dir / "closing_experience.md"
    marker = "## Results of the last slate\n\n"
    text = closing.read_text() if closing.exists() else ""
    if marker in text and names and all(f"`{n}`" in text for n in names):
        out["results_block"] = text[text.index(marker) + len(marker):].strip()
        out["names"] = set()
        return out

    def _line(split: str, s: dict) -> str:
        n = max(s.get("total") or 0, 1)
        return f"- {split}: acc={s.get('accuracy', 0):.1%}, cost/sample = {s.get('total_weighted_cost', 0) / n:.0f}"

    def _blind(c: dict) -> str:
        if paired_pick.blind_direction_only():
            return blind_shown_line(c.get("blind_direction"))
        return _line("blind", c.get("blind_summary") or {})
    rows = [f"### `{c.get('candidate')}` @ `{c.get('anchor')}`\n"
            + _line("feedback", c.get("feedback_summary") or {}) + "\n" + _blind(c)
            for c in cands]
    out["results_block"] = "\n\n".join(rows) or None
    return out


def build_continue_prompt(
    iteration: int,
    candidate_results: str,
    workflow_count: int,
    workflow_summary: str,
    search_results: str,
    requested_cards: str,
    performance_history: str,
    bar_to_beat: str = "",
    show_reference: bool = False,
    run_structure: str = "",
) -> str:
    """Iteration >= 2 in a resumed strategist session: only what is new since
    the last answer (results, the updated bar, this round's retrieval, the
    history) plus the slate rules. The system prompt, the baseline table, the
    reference cells and the benchmark description are already in the session."""
    tpl = _apply_reference_blocks(load_prompt("continue"), show_reference)
    return (
        tpl.replace("{iteration}", str(iteration))
        .replace("{bar_to_beat}", bar_to_beat or "(no bar computed)")
        .replace("{candidate_results}", candidate_results)
        .replace("{workflow_count}", str(workflow_count))
        .replace("{workflow_summary}", workflow_summary)
        .replace("{search_results}", search_results)
        .replace("{requested_cards}", requested_cards)
        .replace("{performance_history}", performance_history or "(none)")
        .replace("{run_structure}", run_structure or "(none yet)")
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_endpoints(spec: str) -> dict[str, str]:
    """Parse "tier=url,tier=url,..." into a dict."""
    out: dict[str, str] = {}
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "=" not in part:
            raise ValueError(f"Bad endpoint spec: {part!r}")
        k, v = part.split("=", 1)
        out[k.strip()] = v.strip()
    for t in TIERS:
        if t not in out:
            raise ValueError(f"Missing endpoint for tier {t}")
    return out


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Strategist: search for an inference strategy on one task")
    p.add_argument("--benchmark", required=True)
    p.add_argument("--val-set", required=True)
    p.add_argument("--iterations", type=int, default=10)
    # Split caps per paper A.3: training/validation/hidden capped at 100
    # samples each; small benchmarks scale down proportionally at load time.
    p.add_argument("--feedback-n", type=int, default=100,
                   help="Training-split cap (strategist sees scores + full traces).")
    p.add_argument("--blind-n", type=int, default=100,
                   help="Validation-split cap (strategist sees aggregate scores only).")
    p.add_argument("--hidden-n", type=int, default=100,
                   help="Hidden-split cap. Meta-agent never sees these "
                        "(not even the aggregate score). Used to pick the "
                        "winner across all iterations, decorrelating selection "
                        "from the signal the meta-agent is optimizing.")
    p.add_argument("--test-n", type=int, default=200,
                   help="Deterministic samples for the final test-set pass on the winner")
    p.add_argument("--run-seed", type=int, default=None,
                   help="Seed for val-set sampling. With seed=N, we shuffle the "
                        "val file deterministically and take the first "
                        "(feedback_n + blind_n + hidden_n) rows. Same seed → "
                        "same split. Use different seeds across parallel runs "
                        "to get different splits. Unset → first N rows in "
                        "file order.")
    p.add_argument("--prompt-variant", default=None,
                   help="Persona name from strategist/prompts/personas/*.md. "
                        "Overrides the run-seed-derived persona slot. "
                        "Options: default, diversity, exploit, tier_ladder, "
                        "multi_only. Unset → derived from run-seed (0-1 default, "
                        "2-3 diversity, 4-5 exploit, 6-7 tier_ladder, 8-9 multi_only).")
    p.add_argument("--out-dir", default=None,
                   help="Override output directory (default: runs/<task>/). "
                        "Use to isolate per-run artifacts in a sweep.")
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument("--per-sample-timeout", type=int, default=0,
                   help="0 = no per-sample timeout (recommended). SLURM walltime is the outer bound.")
    p.add_argument(
        "--endpoints",
        required=True,
        help="Comma-separated base URLs per tier: fast=URL,balanced=URL,complex=URL",
    )
    p.add_argument(
        "--models",
        default=("fast=Qwen/Qwen3-0.6B,balanced=Qwen/Qwen3-1.7B,"
                 "complex=Qwen/Qwen3-4B"),
        help="Comma-separated model names per tier. Default: the paper's "
             "executor tiers Qwen3-0.6B (fast) / 1.7B (balanced) / "
             "Qwen3-4B (complex; both modes).",
    )
    p.add_argument(
        "--strategist", dest="commander",
        choices=sorted(COMMANDERS.keys()),
        default="codex",
        help="The strategist LLM: codex (the Codex CLI, model from STRATEGIST_MODEL), "
             "claude-opus (the Anthropic API, ANTHROPIC_API_KEY) or claude-minimax "
             "(the Claude Code CLI on a MiniMax endpoint, MINIMAX_API_KEY).",
    )
    p.add_argument(
        "--user-hint",
        default=None,
        help="NOT part of the paper's method (the search is automatic). Optional "
             "human-written diagnosis/hint prepended to the strategist prompt. "
             "Use to nudge the orchestrator on stuck benchmarks (e.g. 'baseline uses tool-execution; "
             "your prior runs never tried agent_code-style workflows; propose tool-use candidates').",
    )
    p.add_argument(
        "--description", default=None,
        help="What the task asks for, shown to the strategist: a sentence or two, or the "
             "path of a text file. Default: the task name.",
    )
    p.add_argument(
        "--reference", default=None, metavar="BLUEPRINT@TIER/MODE",
        help="The reference strategy BL, a library blueprint at one anchor, e.g. "
             "chain_of_thought@complex/thinking. The harness measures it in iteration 1, "
             "every candidate is compared with it question by question, and the final pick "
             "is the candidate that beats it on the validation and hidden splits by the "
             "largest hidden margin, or BL itself (Algorithm 1). Without it the final pick "
             "is the highest hidden-set accuracy, ties by lower cost.",
    )
    p.add_argument(
        "--no-library",
        action="store_true",
        help="Ablation: disable all library/experience context (no reference_block, "
             "no inspiration workflows, no card lookups, no search-query expansion). "
             "The commander only sees the task description and sample questions "
             "(plus the baseline bar with --baseline-targets). Combine with "
             "--iterations 1 for a true zero-shot run.",
    )
    p.add_argument(
        "--no-resume-session",
        action="store_true",
        help="Start a new strategist session every call, so each iteration sees only its "
             "own prompt (a stateless strategist). By default a run keeps one session: "
             "iteration 1 sends the full prompt and later iterations send only what is new "
             "(prompts/continue.md).",
    )
    p.add_argument(
        "--blind-feedback",
        choices=("numbers", "direction"),
        default="numbers",
        help="What the strategist sees of the validation (blind) split: each candidate's "
             "accuracy and cost (numbers, the default), or only whether they are higher, "
             "about the same or lower than the reference's (direction; needs --reference).",
    )
    return p


async def main():
    args = build_parser().parse_args()
    # Read by every renderer that could show blind results (paired_pick.blind_direction_only).
    os.environ[paired_pick.BLIND_VIEW_ENV] = args.blind_feedback
    if args.blind_feedback == "direction":
        print("[non-paper] --blind-feedback direction: the strategist sees blind results only "
              "as a direction vs the reference", flush=True)
    # The launcher's `stop` creates this file; the run then shuts itself and its
    # Codex / OpenHands child processes down (strategist/stop_flag.py).
    from strategist import stop_flag
    stop_flag.install(os.environ.get("STRATEGIST_STOP_FILE"), f"strategist {args.benchmark}")

    _use_out_dir(Path(args.out_dir) if args.out_dir else STRATEGIST_DIR.parent / "runs" / args.benchmark)
    _ensure_out_dirs()

    endpoints = _parse_endpoints(args.endpoints)
    model_names = _parse_endpoints(args.models)
    llms = build_llms_dict(endpoints, model_names)

    # Configure module-level default client (some workflows may call
    # generate/chat without passing llm) to balanced/nothink.
    from strategist.basic.generate import set_backend
    set_backend(base_url=endpoints["balanced"], model=model_names["balanced"])

    benchmark = args.benchmark
    bench_desc = args.description or f"Task: {benchmark}"
    if args.description and Path(args.description).is_file():
        bench_desc = Path(args.description).read_text().strip()
    # Val-set split: deterministic and (optionally) seeded per run so parallel
    # sweep runs each see a different 75 samples (but the same 75 every time
    # for that run-id). Test set stays unshuffled for global reproducibility.
    #
    # 3-way split:
    #   feedback  — full transcripts shown to meta-agent (debugging signal)
    #   blind     — score-only shown to meta-agent (its own dev signal)
    #   hidden    — never shown at all; used solely to pick the winner
    #               across all iterations. Decorrelates selection from the
    #               signal the meta-agent is allowed to optimize, preventing
    #               Goodhart-style overfitting to blind acc across 50
    #               candidates (5 candidates × 10 iterations).
    requested_total = args.feedback_n + args.blind_n + args.hidden_n
    # First load everything available (no cap) so we know actual size.
    all_val_full = load_jsonl_deterministic(
        args.val_set,
        limit=0,  # 0 = no limit
        seed=args.run_seed,
    )
    available = len(all_val_full)
    if available < requested_total:
        # Scale each split down proportionally, preserving the ratio the user
        # asked for. Small benchmarks (aime_2024 test=30, humaneval val=32,
        # several at val=100) cannot provide 3×100=300 samples; we adapt
        # rather than silently giving the second/third split zero samples.
        ratio = available / requested_total
        fb_n = max(1, int(args.feedback_n * ratio))
        bl_n = max(1, int(args.blind_n * ratio))
        hd_n = max(1, available - fb_n - bl_n)  # rest
        print(f"[val-adjust] requested {requested_total} samples; only {available} "
              f"available. Scaling to feedback={fb_n}, blind={bl_n}, hidden={hd_n}.")
    else:
        fb_n, bl_n, hd_n = args.feedback_n, args.blind_n, args.hidden_n
    val_total = fb_n + bl_n + hd_n
    all_val = all_val_full[:val_total]
    feedback_samples = all_val[:fb_n]
    blind_samples = all_val[fb_n : fb_n + bl_n]
    hidden_samples = all_val[fb_n + bl_n : val_total]
    print(f"=== Strategist search on {benchmark} ===")
    print(f"Feedback: {len(feedback_samples)}, Blind: {len(blind_samples)}, "
          f"Hidden: {len(hidden_samples)} (never shown to meta-agent)")
    print(f"Iterations: {args.iterations}")

    workflow_count = library_size()

    prev_search_queries: list[str] = []
    prev_card_requests: list[str] = []
    last_candidate_results_block: str | None = None
    # Names evaluated in the previous iteration: each must receive an
    # experience note in the next slate (paper Sec. 3 step 4), enforced below.
    prev_candidate_names: set[str] = set()

    # Track the best (candidate, anchor) across all iterations by blind acc.
    # Tiebreak: lower weighted cost wins. This is what gets the final
    # test-set eval.
    best: dict | None = None

    # Resume support: drive from log.jsonl completions. The last
    # fully-logged iter is whatever max iteration appears in log.jsonl
    # with either candidates evaluated or a terminal error (parse_failure
    # / commander_failure — unrecoverable for that iter). No more orphan
    # candidate files to clean up — the library is our persistent store
    # now and its writes are idempotent (existing files are never
    # overwritten).
    resume_from = 1
    if RUN_LOG.exists():
        max_logged = 0
        for line in RUN_LOG.open():
            try:
                j = json.loads(line)
            except Exception:
                continue
            it = j.get("iteration", 0)
            completed = bool(j.get("candidates")) or bool(j.get("error"))
            if completed and it > max_logged:
                max_logged = it
        if max_logged:
            resume_from = max_logged + 1
            print(f"[resume] last fully-logged iter={max_logged}; starting from iter {resume_from}")

    # Resume support continued: rebuild `best` from log.jsonl so the
    # post-iter test pass has a winner to evaluate even when resume_from
    # already exceeds args.iterations (i.e., all iters completed in a
    # prior run that timed out during the test pass). The winner is now
    # identified by candidate name alone — resolve_workflow() imports it
    # from the library when the test pass runs.
    if resume_from > 1 and RUN_LOG.exists():
        best_acc = -1.0
        for line in RUN_LOG.open():
            try:
                j = json.loads(line)
            except Exception:
                continue
            for c in j.get("candidates", []) or []:
                hds = c.get("hidden_summary") or {}
                bls = c.get("blind_summary") or {}
                fbs = c.get("feedback_summary") or {}
                made_calls = (fbs.get("total_weighted_cost", 0) > 0
                              or bls.get("total_weighted_cost", 0) > 0
                              or hds.get("total_weighted_cost", 0) > 0)
                if not made_calls:
                    continue
                hd_acc = hds.get("accuracy", 0)
                hd_cost = hds.get("total_weighted_cost", 0)
                if best is None or selection_key(fbs, bls, hds) > selection_key(
                        best["feedback_summary"], best["blind_summary"], best["hidden_summary"]):
                    iter_idx = j.get("iteration")
                    best_acc = hd_acc
                    best = {
                        "iteration": iter_idx,
                        "candidate": c.get("candidate"),
                        "anchor": c.get("anchor"),
                        "prompts": c.get("prompts") or {},
                        "hidden_accuracy": hd_acc,
                        "hidden_cost": hd_cost,
                        "hidden_summary": hds,
                        "blind_accuracy": bls.get("accuracy", 0),
                        "blind_cost": bls.get("total_weighted_cost", 0),
                        "blind_summary": bls,
                        "feedback_summary": fbs,
                    }
        if best is not None:
            print(f"[resume] rebuilt best from log: {best['candidate']} @ {best['anchor']} "
                  f"iter{best['iteration']} hidden={best_acc:.1%} cost={best['hidden_cost']:.2e}")
        else:
            print(f"[resume] log had no winner-eligible candidates; best remains None")

    if resume_from > 1 and RUN_LOG.exists():
        restored = restore_round_state(RUN_LOG, RUN_PROMPTS_DIR)
        if restored:
            prev_search_queries = restored["search_queries"]
            prev_card_requests = restored["read_cards"]
            prev_candidate_names = restored["names"]
            last_candidate_results_block = restored["results_block"]
            print(f"[resume] restored iteration {resume_from - 1}'s state: results block "
                  f"{len(last_candidate_results_block or '')} chars, {len(prev_card_requests)} card request(s), "
                  f"{len(prev_candidate_names)} blueprint(s) owing notes")

    persona = resolve_persona(args.run_seed, args.prompt_variant)
    print(f"[persona] run_seed={args.run_seed} persona={persona}")

    # The reference strategy (BL) as (blueprint, anchor): measured in iteration 1,
    # compared with every candidate, and the bar the final pick must beat.
    ref_cell = None
    if args.reference:
        if "@" not in args.reference:
            sys.exit("--reference must be BLUEPRINT@TIER/MODE, e.g. chain_of_thought@complex/thinking")
        ref_cell = tuple(args.reference.split("@", 1))
        print(f"[reference] {ref_cell[0]} @ {ref_cell[1]}: measured in iteration 1; the final pick must "
              f"beat it on the validation and hidden splits")

    def _reference_feedback(cell, split="feedback"):
        """The reference's per-question results on `split` (its first measurement)."""
        pattern = f"iter_*/{cell[0]}__{cell[1].replace('/', '_')}/{split}/samples.jsonl"
        for p in sorted((RUN_DIR / "generations").glob(pattern)):
            q = paired_pick._samples(p)
            if q:
                return q
        return None

    # One strategist session per run (default; --no-resume-session turns it off):
    # every call resumes it, so the strategist keeps its earlier prompts, answers
    # and reasoning. Once the session holds the system prompt and the static
    # context, iteration >= 2 sends only what is new (prompts/continue.md),
    # and a re-ask sends only the correction; without a session every call gets
    # the full prompt.
    import inspect
    commander_fn = COMMANDERS[args.commander]
    session_file = RUN_DIR / "commander_session_id.txt"
    resume_session = (not getattr(args, "no_resume_session", False)
                      and "session_file" in inspect.signature(commander_fn).parameters)
    commander_kw = {"session_file": session_file} if resume_session else {}
    seen_cards_path = RUN_DIR / "commander_session_cards.json"

    def _session_id() -> str:
        return session_file.read_text().strip() if session_file.exists() else ""

    def _seen_cards() -> dict[str, str]:
        """Cards (name -> text as last sent) the live session already holds."""
        try:
            d = json.loads(seen_cards_path.read_text())
        except (OSError, ValueError):
            return {}
        return d.get("cards", {}) if _session_id() and d.get("session") == _session_id() else {}

    print(f"[session] strategist session: {'one per run, resumed every call' if resume_session else 'new every call'}")

    for iteration in range(resume_from, args.iterations + 1):
        print(f"\n--- Iteration {iteration}/{args.iterations} ---")

        # vLLM pods churn across the 1-2hr run; re-validate endpoints each
        # iteration and swap in a live replacement for any dead URL. Without
        # this, stale baked-in endpoints cascade into APIConnectionError on
        # later iterations and the final test eval.
        refresh_llms_dict(llms, model_names)

        if getattr(args, "no_library", False):
            # Ablation: strip every library/experience signal. Commander
            # designs from scratch using only the task, samples, and bar.
            search_results = "(library disabled — no search expansion)"
            inspiration_block = "(library disabled — no inspiration workflows)"
            requested_cards = "(library disabled — no card lookups)"
        else:
            # Paper A.4 retrieval: relevance hits rendered with full cards,
            # adjacent inspiration as one-liners, hybrid-scored and deduped.
            search_results, inspiration_block = run_library_retrieval(
                prev_search_queries, fallback_task=f"{benchmark}: {bench_desc}"
            )
            requested_cards = load_requested_cards(prev_card_requests)
            # cards as seen from this benchmark: its own baseline row, the latest notes
            search_results = cards_for_prompt(search_results, benchmark)
            requested_cards = cards_for_prompt(requested_cards, benchmark)
        system_prompt = build_system_prompt(
            workflow_count, search_results, requested_cards, persona=persona,
            resume_session=resume_session,
        )

        perf_history = render_iteration_history(RUN_LOG, benchmark)
        run_structure = render_run_structure(RUN_LOG)
        # The prompt holds the task description, the retrieved blueprints and the
        # prior-iteration log (paper A.5). With --reference it also carries the
        # reference strategy's measured results: the bar to beat.
        show_targets = ref_cell is not None
        bar = paired_pick.render_reference_bar(ref_cell, RUN_LOG) if ref_cell else ""
        hint_block = ""
        if getattr(args, "user_hint", None):
            print("[non-paper] --user-hint: human-written guidance is injected "
                  "into the strategist prompt", flush=True)
            hint_block = (
                "## Targeted hint (human-written diagnosis of why prior runs stalled)\n\n"
                f"{args.user_hint}\n\n"
                "Take this hint seriously: prior runs failed to fix the issues it describes. "
                "Address them directly in your candidates this iteration.\n\n"
                "---\n\n"
            )
        if last_candidate_results_block is None:
            user_prompt = build_create_prompt(
                benchmark_name=benchmark,
                benchmark_description=bench_desc,
                sample_questions=feedback_samples,
                workflow_count=workflow_count,
                workflow_summary=inspiration_block,
                search_results=search_results,
                requested_cards=requested_cards,
                performance_history=perf_history,
                bar_to_beat=bar,
                show_reference=show_targets,
                run_structure=run_structure,
            )
        else:
            user_prompt = build_revise_prompt(
                iteration=iteration,
                benchmark_name=benchmark,
                benchmark_description=bench_desc,
                candidate_results=last_candidate_results_block,
                workflow_count=workflow_count,
                workflow_summary=inspiration_block,
                search_results=search_results,
                requested_cards=requested_cards,
                performance_history=perf_history,
                bar_to_beat=bar,
                show_reference=show_targets,
                run_structure=run_structure,
            )

        user_prompt = hint_block + user_prompt
        if getattr(args, "no_library", False):
            user_prompt += NO_LIBRARY_NOTE
        elif library_size() == 0:
            user_prompt += EMPTY_LIBRARY_NOTE
        full_prompt = system_prompt + "\n\n---\n\n" + user_prompt
        # Iteration >= 2 in a live session sends only what is new (see above the loop).
        prompt = full_prompt
        if resume_session and last_candidate_results_block is not None and _session_id():
            seen = _seen_cards()
            prompt = hint_block + build_continue_prompt(
                iteration=iteration,
                candidate_results=last_candidate_results_block,
                workflow_count=workflow_count,
                workflow_summary=inspiration_block,
                search_results=dedupe_cards(search_results, seen),
                requested_cards=dedupe_cards(requested_cards, seen),
                performance_history=perf_history,
                bar_to_beat=bar,
                show_reference=show_targets,
                run_structure=run_structure,
            )
            if getattr(args, "no_library", False):
                prompt += NO_LIBRARY_NOTE
            elif library_size() == 0:
                prompt += EMPTY_LIBRARY_NOTE
        session_note = ("resumed session, short prompt" if prompt is not full_prompt
                        else "session " + ("start" if resume_session and not session_file.exists() else
                                           "resume" if resume_session else "off"))
        print(f"  [commander={args.commander}] prompt_len={len(prompt)} "
              f"(full={len(full_prompt)}; {session_note}), calling...")

        # Persist the exact prompt we're sending so we can reproduce the
        # commander call later.
        prompt_path = RUN_PROMPTS_DIR / f"iter_{iteration:03d}.md"
        prompt_path.write_text(prompt)

        # Commander call with rejection loop (paper A.4): a response missing
        # the ```json slate, omitting required candidate fields, missing one
        # of the three kinds, or structurally duplicating a recent candidate
        # is REJECTED and the strategist is re-prompted with the violations.
        # `# WORKFLOW:` python blocks are optional (all-reuse slates are
        # valid). After 3 attempts the iteration is logged as failed.
        raw = None
        meta = None
        candidates_meta: list[dict] = []
        written_names: list[str] = []
        ingest_warnings: list[str] = []
        written_names_run: set[str] = set()
        iter_written: set[str] = set()   # every file written this iteration, all attempts
        last_err = None
        correction_nudge = ""
        raw_path = RUN_RAW_DIR / f"iter_{iteration:03d}.md"
        # Up to 3 rounds. Round 1 sends the full prompt. When the answer fails a
        # check but its slate parsed, the next round is a REPAIR: the strategist
        # gets its own slate back with the violations and returns only the
        # missing pieces (a 5th candidate, a `target`, the missing experience
        # notes, a replacement for a duplicate), which are merged in and
        # re-checked. Only an answer without a usable json block is re-asked
        # from scratch.
        repair: dict | None = None    # {"meta": previous slate, "violations": [...]}
        for attempt in range(3):
            if repair is not None:
                prompt_to_send = build_repair_prompt(repair["meta"], repair["violations"],
                                                     sorted(prev_candidate_names))
            elif resume_session and correction_nudge:
                # the session already holds the prompt and the faulty answer
                prompt_to_send = correction_nudge.lstrip("\n-")
            else:
                prompt_to_send = prompt + correction_nudge
            try:
                raw = commander_fn(prompt_to_send, **commander_kw)
            except Exception as e:
                last_err = f"commander_exception: {e}"
                print(f"  [error] commander attempt {attempt+1}/3 raised: {e}")
                if isinstance(e, CodexSessionLost) and prompt is not full_prompt and repair is None:
                    # the session is gone: a new one needs the whole context
                    print("  [session] resume failed; starting a new session with the full prompt")
                    prompt = full_prompt
                    prompt_path.write_text(prompt)
                continue
            if resume_session and prompt_to_send.startswith(prompt) and _session_id():
                # the session now holds these cards: later rounds point at them
                cards = {**_seen_cards(), **card_bodies(search_results), **card_bodies(requested_cards)}
                seen_cards_path.write_text(json.dumps({"session": _session_id(), "cards": cards}, indent=1))
            # Persist every response verbatim: the first answer as iter_XXX.md,
            # repairs as iter_XXX_repairN.md (the merged slate is logged).
            if attempt == 0:
                raw_path.write_text(raw)
            else:
                (RUN_RAW_DIR / f"iter_{iteration:03d}_repair{attempt}.md").write_text(raw)
                (RUN_PROMPTS_DIR / f"iter_{iteration:03d}_repair{attempt}.md").write_text(prompt_to_send)
            if "```json" not in raw:
                last_err = "missing_json_block"
                print(f"  [warn] commander attempt {attempt+1}/3 returned no ```json block; retrying")
                if repair is None:
                    correction_nudge = (
                        "\n\n---\n\nYour previous response did not contain a ```json "
                        "slate block. The slate is mandatory. Please respond with:\n"
                        "1. Zero or more ```python fenced blocks, each headed by "
                        "`# WORKFLOW: <snake_case_name>`, one per new/composed candidate.\n"
                        "2. A ```json block containing the 5-entry `candidates` slate.\n"
                        "Do not omit the json block."
                    )
                continue

            # Ingest `# WORKFLOW:` blocks into the library BEFORE slate
            # resolution so any name the commander just authored can be
            # imported by the resolver below.
            written_names, ingest_warnings = ingest_new_workflows(raw)
            written_names_run.update(written_names)
            iter_written.update(written_names)
            if written_names:
                print(f"  [library] wrote {len(written_names)} new workflow(s): "
                      f"{', '.join(written_names)}")
            for w in ingest_warnings:
                print(f"  [library-warn] {w}")

            try:
                parsed = parse_slate_or_patch(raw)
            except Exception as e:
                last_err = f"parse_failure: {e}"
                print(f"  [warn] commander attempt {attempt+1}/3 json unparseable: {e}")
                if repair is None:
                    correction_nudge = (
                        f"\n\n---\n\nYour previous ```json slate block failed to parse "
                        f"({e}). Re-send the full response with a valid json slate block."
                    )
                continue
            if repair is not None and "candidates" not in parsed:
                meta = merge_slate_patch(repair["meta"], parsed)
                print(f"  [repair] merged patch: {describe_patch(parsed)}")
            else:
                meta = parsed
            candidates_meta = meta.get("candidates", [])

            no_library = bool(getattr(args, "no_library", False))
            violations = slate_violations(candidates_meta, ingest_warnings,
                                          require_kinds=not no_library and library_size() > 0)
            if prev_candidate_names:
                noted = {e.get("workflow") for e in (meta.get("experience") or [])
                         if isinstance(e, dict) and e.get("observation") and e.get("recommendation")}
                missing_notes = sorted(prev_candidate_names - noted)
                if missing_notes:
                    violations.append(
                        "experience notes missing for last iteration's candidates: "
                        + ", ".join(f"`{n}`" for n in missing_notes)
                        + " — every blueprint that was just measured needs an `experience` entry "
                        "with a concrete observation (its numbers, failure modes) and recommendation"
                    )
            if violations:
                last_err = "slate_rejected: " + "; ".join(violations[:3])
                for v in violations:
                    print(f"  [slate-reject] {v}")
                repair = {"meta": meta, "violations": violations}
                meta = None
                continue
            break
        if meta is None:
            print(f"  [error] commander failed after 3 attempts: {last_err}")
            # Files written by the rejected attempts were never measured: they
            # do not belong in the library (every library entry carries a card).
            removed = remove_orphan_blueprints(iter_written, keep=set())
            written_names_run -= set(removed)
            append_jsonl(RUN_LOG, {
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "benchmark": benchmark,
                "iteration": iteration,
                "commander": args.commander,
                "prompt_path": str(prompt_path),
                "error": f"commander_failure_after_retry: {last_err}",
                "raw": (raw or "")[:2000],
            })
            continue

        prev_search_queries = meta.get("search_queries", []) or []
        prev_card_requests = meta.get("read_cards", []) or []

        # Files written by earlier, rejected attempts of this iteration that the
        # accepted slate neither names nor imports were never measured: drop them.
        keep_names = {c.get("name") for c in candidates_meta if isinstance(c, dict)}
        for c in candidates_meta:
            if isinstance(c, dict):
                keep_names.update(c.get("builds_on") or [])
        removed = remove_orphan_blueprints(iter_written, keep=keep_names)
        written_names_run -= set(removed)
        written_names = sorted(iter_written - set(removed))
        # Sidecar per new blueprint: the task it was designed for. Retrieval
        # indexes it and the cards show it, so a task-specific blueprint is not
        # mistaken for a general one on another benchmark.
        try:
            from strategist.workflows._qa_common import task_kind
            LIBRARY_META_DIR.mkdir(exist_ok=True)
            by_name = {c.get("name"): c for c in candidates_meta if isinstance(c, dict)}
            for name in written_names:
                c = by_name.get(name, {})
                (LIBRARY_META_DIR / f"{name}.json").write_text(json.dumps({
                    "designed_for": benchmark, "task_kind": task_kind(feedback_samples[0].get("task", "")) if feedback_samples else "",
                    "kind": c.get("kind"), "target": c.get("target"), "builds_on": c.get("builds_on") or [],
                    "run": f"{args.commander}/seed{args.run_seed}", "iteration": iteration,
                    "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
                }, indent=1) + "\n")
        except Exception as exc:
            print(f"  [library-meta] could not write sidecars: {exc}")

        # Record the strategist's narrative experience notes (library writeback).
        try:
            from strategist.experience import record_experience
            commander_id = f"{args.commander}_{benchmark}"
            kept = skipped = 0
            for exp in meta.get("experience", []) or []:
                if (
                    isinstance(exp, dict)
                    and exp.get("workflow")
                    and exp.get("observation")
                    and exp.get("recommendation")
                ):
                    record_experience(
                        workflow_name=exp["workflow"],
                        commander_id=commander_id,
                        task_type=benchmark,
                        observation=exp["observation"],
                        recommendation=exp["recommendation"],
                        accuracy_impact=exp.get("accuracy_impact", ""),
                    )
                    kept += 1
                else:
                    skipped += 1
            if kept or skipped:
                print(f"  [experience] recorded {kept}, rejected {skipped} incomplete")
            # The note arrives one iteration after the measurement: refresh
            # the cards so they show it (seed cards keep their matrix table).
            from strategist.cards import write_measurement_card
            for name in {e.get("workflow") for e in (meta.get("experience") or []) if isinstance(e, dict)}:
                if name and (STRATEGIST_WORKFLOWS_DIR / f"{name}.py").exists():
                    try:
                        write_measurement_card(name)
                    except Exception:
                        pass
        except Exception as exp_err:
            print(f"  [experience] record failure: {exp_err}")

        # Audit the slate (non-fatal: name format, builds_on resolvability).
        # These warnings do not kill the run — they are fed into the next
        # iteration's prompt (paper A.4).
        slate_warnings = audit_slate(
            candidates_meta, written_names_run, ingest_warnings
        )
        meta["_slate_warnings"] = slate_warnings
        for w in slate_warnings:
            print(f"  [slate-warn] {w}")
        print(f"  [parse] {len(candidates_meta)} candidates declared "
              f"(library grew by {len(written_names)} this iter)")

        # 4) Resolve every (candidate, anchor) pair against the library.
        #    Unresolved names are dropped with a note fed back next iter.
        jobs: list[tuple[str, str, str, Any, Any]] = []
        skipped_blocks: list[str] = []
        valid_anchors = {"multi"} | {f"{t}/{m}" for t in TIERS for m in MODES}
        for spec in candidates_meta[:5]:
            cand_name = spec.get("name", "")
            anchors = (spec.get("tier_mode_anchors") or ["multi"])[:3]
            why = spec.get("why", "")
            prompts = spec.get("prompts") or None
            fn, err = resolve_strategy(cand_name, prompts)
            if fn is None:
                skipped_blocks.append(
                    f"### `{cand_name}` — skipped ({err})"
                )
                continue
            for anchor in anchors:
                if anchor not in valid_anchors:
                    skipped_blocks.append(
                        f"### `{cand_name}` @ `{anchor}` — skipped (invalid anchor)"
                    )
                    continue
                jobs.append((cand_name, anchor, why, fn, prompts))

        # The reference baseline (the matrix's top cell, run with its exact
        # blueprint) must be measured in this run: the final pick compares every
        # candidate with it question by question (strategist/paired_pick.py).
        if ref_cell and iteration == 1 and not any(
                (j[0], j[1]) == ref_cell and not j[4] for j in jobs):
            fn, err = resolve_strategy(ref_cell[0], None)
            if fn is not None:
                jobs.append((ref_cell[0], ref_cell[1],
                             "reference: the top baseline cell, run exactly as in the baseline table "
                             "(added by the harness)", fn, None))
                print(f"  [reference] added {ref_cell[0]} @ {ref_cell[1]} to the slate")
            else:
                print(f"  [reference] could not resolve {ref_cell[0]}: {err}")

        print(f"  [eval] dispatching {len(jobs)} (candidate × anchor) pairs × (feedback + blind + hidden) in parallel...")

        # Per-run eval cache: skip re-evaluating the same (workflow, anchor)
        # cell if it was already measured earlier this run. Within a run the
        # feedback/blind/hidden sample sets are fixed (deterministic split
        # by run_seed), so cached summaries are exactly what a re-eval
        # would produce. Across runs the sample sets differ, which is why
        # the cache file is per-out-dir, not global.
        eval_cache = _load_eval_cache()

        async def _run_pair(cand_name: str, anchor: str, why: str, fn, prompts=None):
            wf_hash = _workflow_content_hash(cand_name)
            if wf_hash and prompts:
                # The same blueprint with other prompts is another strategy.
                wf_hash = f"{wf_hash}+{overrides_key(prompts)}"
            hit = eval_cache.get((wf_hash, anchor)) if wf_hash else None
            if hit is not None:
                # Synthesize the same {summary, samples} shape eval_candidate
                # returns. The commander only reads summary; `samples: []` is
                # safe because render_candidate_results handles it.
                def _restore(kind: str) -> dict:
                    return {"summary": {**hit.get(f"{kind}_summary", {}), "cached_from_iteration": hit.get("iteration")},
                            "samples": []}
                print(f"  [cache-hit] {cand_name} @ {anchor} — reusing cached summaries from iter {hit.get('iteration','?')}")
                return cand_name, anchor, why, _restore("feedback"), _restore("blind"), _restore("hidden"), prompts

            rec_root = RUN_DIR / "generations" / f"iter_{iteration:02d}" / paired_pick.gen_dir_name(cand_name, anchor, prompts)
            fb_task = eval_candidate(
                workflow_fn=fn, samples=feedback_samples, llms=llms,
                anchor=anchor, expected_field="answer",
                concurrency=args.concurrency,
                per_sample_timeout=args.per_sample_timeout,
                record_dir=rec_root / "feedback",
            )
            bl_task = eval_candidate(
                workflow_fn=fn, samples=blind_samples, llms=llms,
                anchor=anchor, expected_field="answer",
                concurrency=args.concurrency,
                per_sample_timeout=args.per_sample_timeout,
                record_dir=rec_root / "blind",
            )
            hd_task = eval_candidate(
                workflow_fn=fn, samples=hidden_samples, llms=llms,
                anchor=anchor, expected_field="answer",
                concurrency=args.concurrency,
                per_sample_timeout=args.per_sample_timeout,
                record_dir=rec_root / "hidden",
            )
            fb, bl, hd = await asyncio.gather(fb_task, bl_task, hd_task)
            # Write-through: record the result so a future iter picking the
            # same cell gets a cache hit. Only cache cells that actually ran
            # (nonzero cost) — a 0-cost cell is evidence of a transient
            # failure, not a real measurement, and caching it would poison
            # the next iter's decision.
            made_calls = (
                (fb["summary"].get("total_weighted_cost", 0) or 0) > 0
                or (bl["summary"].get("total_weighted_cost", 0) or 0) > 0
                or (hd["summary"].get("total_weighted_cost", 0) or 0) > 0
            )
            if wf_hash and made_calls:
                _append_eval_cache({
                    "workflow_name": cand_name,
                    "workflow_hash": wf_hash,
                    "anchor": anchor,
                    "iteration": iteration,
                    "feedback_summary": fb["summary"],
                    "blind_summary": bl["summary"],
                    "hidden_summary": hd["summary"],
                })
                eval_cache[(wf_hash, anchor)] = {
                    "iteration": iteration,
                    "feedback_summary": fb["summary"],
                    "blind_summary": bl["summary"],
                    "hidden_summary": hd["summary"],
                }
            return cand_name, anchor, why, fb, bl, hd, prompts

        pair_results = await asyncio.gather(*(_run_pair(*j) for j in jobs))

        result_blocks: list[str] = list(skipped_blocks)
        iter_log: list[dict] = []
        for cand_name, anchor, why, fb, bl, hd, prompts in pair_results:
            fbs, bls, hds = fb["summary"], bl["summary"], hd["summary"]
            print(
                f"  [done] {cand_name} @ {anchor} — "
                f"feedback acc={fbs['accuracy']:.1%} errors={fbs['errors']} cost={fbs['total_weighted_cost']:.2e} | "
                f"blind acc={bls['accuracy']:.1%} errors={bls['errors']} cost={bls['total_weighted_cost']:.2e} | "
                f"hidden acc={hds['accuracy']:.1%} errors={hds['errors']} (selection signal — not shown to meta-agent)"
            )
            # If the candidate is effectively broken (every set scored 0% AND
            # there were errors), print the top error types so we can diagnose
            # without digging into sample logs.
            if fbs['accuracy'] == 0 and bls['accuracy'] == 0 and hds['accuracy'] == 0:
                for label, s in (("feedback", fbs), ("blind", bls), ("hidden", hds)):
                    tops = s.get("top_errors") or []
                    if tops:
                        err_str = "; ".join(f"{e['count']}× {e['msg']}" for e in tops)
                        print(f"    [0%-diagnose/{label}] {err_str}")
                        break
            # render_candidate_results() emits feedback + blind only — the
            # hidden summary is intentionally excluded from anything that
            # reaches the meta-agent prompt.
            shown_why = why
            if prompts:
                shown_why = f"{why} [prompt slots replaced: {', '.join(sorted(prompts))}]"
            # Direction-only runs: the blind line is this candidate vs the
            # reference on the same blind questions, without numbers.
            bdir = None
            if ref_cell and (cand_name, anchor) == ref_cell and not prompts:
                bdir = "this is the reference"
            elif ref_cell:
                ref_bl = _reference_feedback(ref_cell, "blind")
                bq = {i: (float(x.get("score") or 0.0), float(x.get("weighted_cost") or 0.0))
                      for i, x in enumerate(bl.get("samples") or [])}
                bdir = paired_pick.direction(bq, ref_bl) if ref_bl and bq else None
            if bdir is None and ref_cell and bls.get("cached_from_iteration") is not None:
                # served from the eval cache (no samples in memory): its rows on disk
                gen = (RUN_DIR / "generations" / f"iter_{int(bls['cached_from_iteration']):02d}"
                       / paired_pick.gen_dir_name(cand_name, anchor, prompts))
                cq = paired_pick._samples(gen / "blind" / "samples.jsonl")
                ref_bl = _reference_feedback(ref_cell, "blind")
                bdir = paired_pick.direction(cq, ref_bl) if cq and ref_bl else None
            block = render_candidate_results(cand_name, anchor, fb, bl, shown_why, blind_direction=bdir)
            if ref_cell and ((cand_name, anchor) != ref_cell or prompts):
                ref_fb = _reference_feedback(ref_cell)
                line = paired_pick.vs_reference_line(ref_fb, fb.get("samples") or [])
                if line:
                    block = block.replace("\nper-tier call breakdown", f"\n{line}per-tier call breakdown", 1)
            result_blocks.append(block)
            iter_log.append({
                "candidate": cand_name,
                "anchor": anchor,
                "why": why,
                "prompts": prompts or {},  # task prompts stay with the run, not the library
                "feedback_summary": fbs,
                "blind_summary": bls,
                "blind_direction": bdir,  # what a direction-only run shows of blind
                "hidden_summary": hds,   # logged on disk only, not shown
            })

            # Feed the measurement back into the shared library experience
            # store so future iterations (any benchmark, any run) see it.
            try:
                from strategist.experience import record_measurement
                # Paper A.3: the shared library store carries training and
                # validation signal only. Hidden-set numbers stay in the run log
                # (selection needs them) and must not leak into experience,
                # cards, or retrieval — across iterations or to future tasks.
                # A re-listed candidate served from the eval cache already has
                # its measurement on record: do not duplicate the row.
                if fbs.get("cached_from_iteration") is None:
                    record_measurement(
                        workflow_name=cand_name,
                        benchmark=benchmark,
                        anchor=anchor,
                        feedback_acc=fbs.get("accuracy", 0.0),
                        feedback_cost=fbs.get("total_weighted_cost", 0.0),
                        blind_acc=bls.get("accuracy", 0.0),
                        blind_cost=bls.get("total_weighted_cost", 0.0),
                        hidden_acc=None,
                        hidden_cost=None,
                        iteration=iteration,
                        run_id=args.run_seed,
                        commander=args.commander,
                        feedback_n=int(fbs.get("total", 0) or 0) or None,
                        blind_n=int(bls.get("total", 0) or 0) or None,
                    )
            except Exception as mx:
                print(f"    [measurement-record] failed for {cand_name}@{anchor}: {mx}")

            # Accumulate (paper Sec. 3 step 4): refresh this blueprint's
            # measurement card so future retrieval — this run or any other
            # task — sees its measured behavior alongside experience notes.
            try:
                from strategist.cards import write_measurement_card
                write_measurement_card(cand_name)
            except Exception as cx:
                print(f"    [card] update failed for {cand_name}: {cx}")

            hd_acc = hds["accuracy"]
            hd_cost = hds["total_weighted_cost"]
            bl_acc = bls["accuracy"]
            challenger = {
                "iteration": iteration,
                "candidate": cand_name,
                "anchor": anchor,
                "prompts": prompts or {},
                "hidden_accuracy": hd_acc,
                "hidden_cost": hd_cost,
                "hidden_summary": hds,
                "blind_accuracy": bl_acc,
                "blind_cost": bls["total_weighted_cost"],
                "blind_summary": bls,
                "feedback_summary": fbs,
            }
            # Winner is picked by HIDDEN accuracy — the signal the meta-agent
            # has never seen. Decorrelates selection from the optimization
            # target, eliminating blind-set Goodhart overfitting across the
            # 50 candidates proposed over 10 iterations.
            #
            # Gate: a candidate must have actually called an LLM on at least
            # one split (nonzero cost) to be eligible. On short-val
            # benchmarks the hidden split can be empty, which would otherwise
            # let a no-op / cheating workflow win by default (first-seen with
            # hd_acc=0 becomes best, nothing beats 0). Seen on aime_2024_8
            # where strict_python_agent_complex "won" with cost=0 and then
            # produced TEST=0/30 cost=0. Per CLAUDE.md, total_weighted_cost
            # must be > 0 on a real result.
            made_calls = (fbs.get("total_weighted_cost", 0) > 0
                          or bls.get("total_weighted_cost", 0) > 0
                          or hds.get("total_weighted_cost", 0) > 0)
            if not made_calls:
                print(f"    [skip-as-winner] {cand_name} @ {anchor} — "
                      f"zero cost on all splits (no LLM calls)")
            elif best is None:
                best = challenger
                print(f"    [BEST so far] {cand_name} @ {anchor} — "
                      f"hidden={hd_acc:.1%} (blind={bl_acc:.1%}, cost={hd_cost:.2e})")
            else:
                # Winner selection per paper Alg. 1 / Sec. 3: the candidate
                # with the highest hidden-set accuracy across ALL iterations
                # wins; ties broken by lower hidden weighted cost. The hidden
                # split is never shown to the strategist, so this selection is
                # decorrelated from the signal it optimizes.
                best_acc = best["hidden_accuracy"]
                best_cost = best.get("hidden_cost", float("inf"))
                rule = selection_rule(hds)
                if selection_key(fbs, bls, hds) > selection_key(
                        best["feedback_summary"], best["blind_summary"], best["hidden_summary"]):
                    best = challenger
                    print(f"    [BEST so far] {cand_name} @ {anchor} — "
                          f"hidden={hd_acc:.1%} (blind={bl_acc:.1%}, cost={hd_cost:.2e}) [rule: {rule}]")
                else:
                    print(f"    [kept current best] {cand_name} @ {anchor} — "
                          f"hidden={hd_acc:.1%} cost={hd_cost:.2e} "
                          f"(best hidden={best_acc:.1%} cost={best_cost:.2e}) [rule: {rule}]")

        # Slate audit warnings are non-fatal but must reach the strategist:
        # they are appended to the results block the next iteration reads.
        if slate_warnings:
            result_blocks.append(
                "### Slate audit warnings (non-fatal — correct these next iteration)\n"
                + "\n".join(f"- {w}" for w in slate_warnings)
            )

        last_candidate_results_block = "\n\n".join(result_blocks) or "(no candidates produced results)"
        prev_candidate_names = {c.get("candidate") for c in iter_log if c.get("candidate")}

        append_jsonl(RUN_LOG, {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "benchmark": benchmark,
            "iteration": iteration,
            "commander": args.commander,
            "prompt_path": str(prompt_path),
            "raw_path": str(raw_path),
            "library_written": written_names,
            "candidates_meta": candidates_meta,
            "candidates": iter_log,
            "search_queries": prev_search_queries,
            "read_cards": prev_card_requests,
            "experience": meta.get("experience") or [],
        })

    # === Closing experience pass: the last slate's blueprints were measured
    # but no later iteration writes about them; ask the strategist for their
    # notes now so every library entry carries at least one (paper Sec. 3).
    if prev_candidate_names and last_candidate_results_block:
        try:
            closing_experience_pass(
                benchmark, sorted(prev_candidate_names), last_candidate_results_block,
                commander_fn, args.commander, RUN_PROMPTS_DIR, RUN_RAW_DIR, RUN_LOG,
                commander_kw=commander_kw,
            )
        except Exception as exc:
            print(f"  [closing-experience] failed: {exc}")

    # === Final pick: with a reference baseline, the paired comparison of
    # strategist/paired_pick.py replaces Alg. 1's highest hidden accuracy (kept in
    # the test results as `alg1_winner` for comparison).
    selection = None
    alg1_best = best
    if ref_cell:
        try:
            selection = paired_pick.choose(RUN_DIR, ref_cell[0], ref_cell[1])
        except Exception as exc:
            print(f"[final-pick] paired pick failed ({type(exc).__name__}: {exc}); falling back to Alg. 1")
    if selection:
        p = selection["pick"]
        entry = None
        for line in RUN_LOG.open():
            try:
                j = json.loads(line)
            except Exception:
                continue
            if j.get("iteration") != p["iteration"]:
                continue
            for c in j.get("candidates") or []:
                if (c.get("candidate"), c.get("anchor"), c.get("prompts") or {}) == (p["candidate"], p["anchor"], p["prompts"] or {}):
                    entry = entry or c
        if entry is None:
            print("[final-pick] the paired pick is not in the log; falling back to Alg. 1")
            selection = None
        else:
            hds, bls, fbs = entry.get("hidden_summary") or {}, entry.get("blind_summary") or {}, entry.get("feedback_summary") or {}
            best = {"iteration": p["iteration"], "candidate": p["candidate"], "anchor": p["anchor"],
                    "prompts": p["prompts"] or {}, "hidden_accuracy": hds.get("accuracy", 0),
                    "hidden_cost": hds.get("total_weighted_cost", 0), "hidden_summary": hds,
                    "blind_accuracy": bls.get("accuracy", 0), "blind_cost": bls.get("total_weighted_cost", 0),
                    "blind_summary": bls, "feedback_summary": fbs}
            print(f"\n=== Final pick (paired vs reference {ref_cell[0]} @ {ref_cell[1]}): {selection['route']} "
                  f"-> {p['candidate']} @ {p['anchor']} (iter {p['iteration']})")
            for s in selection["shortlist"]:
                print(f"    shortlist {s['candidate']} @ {s['anchor']} iter{s['iteration']}: "
                      f"blind d={s['blind']['d'] * 100:+.1f}pp cost {s['blind']['rho']:.2f}x [{s['blind_route']}] | "
                      f"hidden d={s['hidden']['d'] * 100:+.1f}±{s['hidden']['se_d'] * 100:.1f}pp "
                      f"cost {s['hidden']['rho']:.2f}±{s['hidden']['se_rho']:.2f}x [{s.get('hidden_route')}]")
    final_rule = "paired_vs_reference" if selection else (
        selection_rule(best["hidden_summary"]) if best is not None else None)

    # === Final test-set pass on the picked (candidate, anchor) ===
    if best is None:
        print("\n=== No successful candidate — skipping test eval ===")
    else:
        test_path = STRATEGIST_DIR / "data" / f"{benchmark}_test.jsonl"
        if not test_path.exists():
            print(f"\n=== Best: {best['candidate']} @ {best['anchor']} "
                  f"blind={best['blind_accuracy']:.1%} — no test file at {test_path}, skipping ===")
        else:
            print(f"\n=== Final test eval ===")
            print(f"  winner: {best['candidate']} @ {best['anchor']} "
                  f"(iter {best['iteration']}, rule {final_rule}, "
                  f"hidden acc={best['hidden_accuracy']:.1%}, "
                  f"blind acc={best['blind_accuracy']:.1%}, "
                  f"cost={best['hidden_cost']:.2e})")
            # Refresh stale endpoints before the test pass: the 3 vLLM URLs
            # baked in at job start may have died
            # during the 1-2hr iteration loop. Without this, test eval fails
            # with APIConnectionError on all 200 samples.
            refresh_llms_dict(llms, model_names)
            # Deterministic: first N rows of the test file, every run.
            test_samples = load_jsonl_deterministic(str(test_path), args.test_n)
            print(f"  running on {len(test_samples)} test samples (deterministic)...")
            winner_fn, resolve_err = resolve_strategy(best["candidate"], best.get("prompts"))
            if winner_fn is None:
                print(f"  [error] best candidate `{best['candidate']}` "
                      f"could not be resolved from library: {resolve_err}")
            else:
                test_eval = await eval_candidate(
                    workflow_fn=winner_fn,
                    samples=test_samples,
                    llms=llms,
                    anchor=best["anchor"],
                    expected_field="answer",
                    concurrency=args.concurrency,
                    per_sample_timeout=args.per_sample_timeout,
                    record_dir=RUN_DIR / "generations" / "test"
                    / paired_pick.gen_dir_name(best["candidate"], best["anchor"], best.get("prompts")),
                )
                ts = test_eval["summary"]
                print(f"  TEST: acc={ts['accuracy']:.1%} ({ts['passed']}/{ts['total']}) "
                      f"errors={ts['errors']} cost={ts['total_weighted_cost']:.2e}")
                test_out_dir = RUN_DIR / "test_results"
                test_out_dir.mkdir(parents=True, exist_ok=True)
                test_out = test_out_dir / f"{benchmark}_{args.commander}.json"
                test_out.write_text(json.dumps({
                    "benchmark": benchmark,
                    "commander": args.commander,
                    "winner": {
                        "candidate": best["candidate"],
                        "anchor": best["anchor"],
                        "iteration": best["iteration"],
                        "prompts": best.get("prompts") or {},
                        "selection_rule": final_rule,
                    },
                    # the paired pick's evidence (reference, shortlist, route) and what
                    # Alg. 1 (highest hidden accuracy) would have picked instead
                    "selection": selection,
                    "reference_cell": ({"blueprint": ref_cell[0], "anchor": ref_cell[1]} if ref_cell else None),
                    "alg1_winner": ({k: alg1_best.get(k) for k in ("candidate", "anchor", "iteration", "prompts",
                                                                   "hidden_accuracy", "hidden_cost")}
                                    if alg1_best else None),
                    "hidden_summary": best["hidden_summary"],
                    "blind_summary": best["blind_summary"],
                    "feedback_summary": best["feedback_summary"],
                    "test_summary": ts,
                    "test_samples": test_eval["samples"],
                }, indent=2))
                append_jsonl(RUN_LOG, {
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                    "benchmark": benchmark,
                    "commander": args.commander,
                    "final_test": True,
                    "winner": {
                        "candidate": best["candidate"],
                        "anchor": best["anchor"],
                        "iteration": best["iteration"],
                        "prompts": best.get("prompts") or {},
                    },
                    "hidden_summary": best["hidden_summary"],
                    "blind_summary": best["blind_summary"],
                    "test_summary": ts,
                })
                print(f"  saved: {test_out}")

    print(f"\n=== Done: {benchmark} — log: {RUN_LOG} ===")


if __name__ == "__main__":
    asyncio.run(main())
