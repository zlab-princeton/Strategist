"""Agent: multi-step LLM with tool use, powered by OpenHands.

Implementation notes
--------------------
The orchestrator and OpenHands share a single Python 3.12 venv (the
repo-root .venv, or STRATEGIST_OPENHANDS_PYTHON). We still spawn a subprocess
for each agent call — not for env isolation, but so we can wrap it in
``unshare --net --map-root-user`` to sandbox the TerminalTool from the
network (see network-isolation note below). The subprocess gets a JSON
payload on stdin.
The runner (_openhands_runner.py) builds an OpenHands Agent + Conversation
with TerminalTool + FileEditorTool ONLY (no browser/search/MCP) and
returns the final agent text plus token usage.

Network isolation: we attempt to wrap the subprocess in
``unshare --net --map-root-user`` so the sandboxed TerminalTool cannot
reach the internet. BUT this also blocks the orchestrator->vLLM call that
OpenHands itself makes (since vLLM is on a different host). If unshare
breaks that path we fall back to running WITHOUT network isolation and
rely on the cluster-level firewall + the fact that Qwen3-4B has no tool
to exfiltrate data. STRATEGIST_AGENT_NO_SANDBOX controls it:
  unset  the sandbox is used only if the model server is still reachable
         from inside it (probed once per endpoint); otherwise the fallback
         above applies and a notice is printed. Before 2026-10-05 the
         fallback was not implemented: with the variable unset every agent
         call failed with a connection error and returned the error text as
         its answer (zero calls, zero cost, graded wrong). Every earlier
         agent launch set the variable to 1 for that reason.
  1      never use the sandbox
  0      always use the sandbox
"""
from __future__ import annotations

import asyncio
import json
import os
import shutil
import signal
import subprocess
import sys
import threading
from pathlib import Path

META = {
    "name": "agent",
    "description": "Multi-step LLM with tool use (OpenHands). The model can run terminal commands and edit files, see results, and iterate until it solves the task.",
    "inputs": {
        "messages": {
            "type": "list[dict]",
            "required": True,
            "description": "Conversation history. The last user message is treated as the task; prior turns are concatenated as context.",
        },
        "tools": {
            "type": "list[str]",
            "required": False,
            "default": ["terminal", "file_editor"],
            "description": "Available tools. Supported: 'terminal' (shell), 'file_editor' (read/write files), and 'python' (alias for terminal). Browser/search are forbidden.",
        },
        "system": {
            "type": "str",
            "required": False,
            "default": "",
            "description": "System instruction prepended to the task.",
        },
        "reasoning": {
            "type": "bool",
            "required": False,
            "default": False,
            "description": "Enable step-by-step reasoning. Forwards to Qwen3 thinking-mode toggle.",
        },
        "temperature": {
            "type": "float",
            "required": False,
            "default": 0.0,
            "description": "Sampling temperature for every LLM call the agent makes.",
        },
        "max_tokens": {
            "type": "int",
            "required": False,
            "default": 8192,
            "description": "Max output tokens per LLM call. Never set below 2048 for Qwen3-Thinking.",
        },
        "max_steps": {
            "type": "int",
            "required": False,
            "default": 10,
            "description": "Maximum agent iterations (tool-use steps) before termination.",
        },
    },
    "output": "str — the agent's final response text after tool use completes",
    "tags": ["basic", "multi-turn", "tool-use", "agent", "openhands"],
}

# Path to the py3.12 interpreter that has openhands-sdk installed: the repo's
# .venv, shared with the orchestrator. Without a .venv at the repo root the
# orchestrator's own interpreter is used (it is the same environment by
# design); a missing interpreter used to make every agent call return an
# error text as its answer.
_REPO_VENV_PYTHON = Path(__file__).resolve().parents[2] / ".venv" / "bin" / "python"
_OPENHANDS_PYTHON = os.environ.get("STRATEGIST_OPENHANDS_PYTHON") or (
    str(_REPO_VENV_PYTHON) if _REPO_VENV_PYTHON.exists() else sys.executable
)

# Path to the runner module file (we invoke it directly by path so the
# subprocess doesn't need to have `strategist` on PYTHONPATH, though we
# add it anyway for safety).
_RUNNER_PATH = str(Path(__file__).resolve().parent / "_openhands_runner.py")

# Repo root — added to subprocess PYTHONPATH so `from strategist...` works
# if the runner ever needs it.
_REPO_ROOT = str(Path(__file__).resolve().parents[2])

# --- Global subprocess-concurrency cap -----------------------------------
# Each OpenHands subprocess imports the openhands-sdk (~500MB-1GB resident)
# AND makes HTTP calls to vLLM. Two independent failure modes at high conc:
#  (1) OOM: at 512 concurrent × 600MB ≈ 300GB, exceeding 192G cgroup limit
#      → memory thrashing, processes stall before first LLM call.
#  (2) CPU/vLLM saturation: 380 runners observed on a 16-CPU orch node → each
#      gets ~4% CPU → imports stretch from ~10s to minutes; meanwhile 380
#      concurrent HTTP calls queue on the per-model vLLM endpoints → LLM
#      turnaround balloons past 900s sample budget. 11,176/12,400 (90%) of
#      agent_code nothink samples timed out with 0 LLM calls under MAX=512.
# Lower default: 32 gives ~2 subprocs/CPU (reasonable for IO-bound agents)
# and ~2 agent calls per vLLM replica when cells are running in parallel.
# Tune higher via STRATEGIST_AGENT_MAX_SUBPROCS if you've verified vLLM + RAM
# headroom. Tune lower for slower hardware.
# Paper A.4: subprocess concurrency is capped at 48.
_AGENT_MAX_SUBPROCS = int(os.environ.get("STRATEGIST_AGENT_MAX_SUBPROCS", "48") or "48")
_agent_subproc_sem: asyncio.Semaphore | None = None


def _get_subproc_sem() -> asyncio.Semaphore:
    """Lazily bind the semaphore to the running event loop."""
    global _agent_subproc_sem
    if _agent_subproc_sem is None:
        _agent_subproc_sem = asyncio.Semaphore(_AGENT_MAX_SUBPROCS)
    return _agent_subproc_sem


def _resolve_base_url_and_model(llm=None) -> tuple[str, str, bool]:
    """Grab endpoint/model/thinking flag.

    Priority:
      1. If ``llm`` is a run_all_cells LB client (has ``_bp_model_name`` +
         ``_bp_endpoints_getter``), use it — this routes correctly per-model
         in the multi-model orchestrator.
      2. Otherwise fall back to the module-level generate._client + _model.
    """
    # Note: `from strategist.basic import generate` would import the FUNCTION
    # (re-exported in __init__.py) not the module — and even `import
    # strategist.basic.generate as _gen` binds the function because of the
    # re-export. Use importlib to get the real module object.
    import importlib
    _gen = importlib.import_module("strategist.basic.generate")

    # Per-client thinking override (multi-tier): when a client
    # carries `_bp_enable_thinking`, use that; otherwise fall back to the
    # module-level default. Back-compat: absent attr → module default.
    if llm is not None and hasattr(llm, "_bp_enable_thinking"):
        enable_thinking = bool(llm._bp_enable_thinking)
    else:
        enable_thinking = bool(_gen._enable_thinking)

    # Case 1: orchestrator LB client
    if llm is not None and hasattr(llm, "_bp_model_name") and hasattr(llm, "_bp_endpoints_getter"):
        eps = llm._bp_endpoints_getter() or []
        if not eps:
            raise RuntimeError(
                f"No healthy endpoints for model {llm._bp_model_name}"
            )
        # Picking eps[0] always routed every agent call to the first registry
        # entry, saturating one pod while peers sat idle (observed 99.3% KV
        # cache on pod[0] with 2250 queued reqs while pods[1..3] were 0%).
        # Random choice spreads load across all healthy endpoints.
        import random as _random
        base_url = str(_random.choice(eps)).rstrip("/")
        return base_url, llm._bp_model_name, enable_thinking

    # Case 1b: multi-tier client — has `_bp_model_name` + own `base_url`
    # but no LB endpoint-getter (single-endpoint pool). Use the client's
    # base_url directly.
    if (
        llm is not None
        and hasattr(llm, "_bp_model_name")
        and not hasattr(llm, "_bp_endpoints_getter")
        and hasattr(llm, "base_url")
    ):
        base_url = str(llm.base_url).rstrip("/")
        return base_url, llm._bp_model_name, enable_thinking

    # Case 2: module-level default client. This path is a safety fallback
    # ONLY for ad-hoc scripts that call agent() without a load-balanced
    # client. In the orchestrator (run_all_cells.py) it indicates a bug —
    # the per-cell traced client did not forward _bp_model_name, so all
    # agent calls would silently route to module-level model (= models[0],
    # typically Qwen/Qwen3-0.6B). Loud-warn so it's caught fast.
    if llm is not None and not (hasattr(llm, "_bp_model_name") and hasattr(llm, "_bp_endpoints_getter")):
        import sys as _sys
        print(
            "[agent.py WARN] llm passed to agent() lacks _bp_model_name/_bp_endpoints_getter. "
            "Falling back to module-level client — this misroutes 1.7B/4B cells to 0.6B in "
            "multi-model orchestrators. Forward LB attrs from worker_client to traced clients.",
            file=_sys.stderr, flush=True,
        )
    client = _gen._client
    if client is None:
        raise RuntimeError(
            "No LLM backend configured. Call strategist.basic.generate.set_backend() first."
        )
    base_url = str(client.base_url).rstrip("/")
    model = _gen._model
    return base_url, model, enable_thinking


def _messages_to_task(messages: list[dict]) -> str:
    """Flatten a chat-style message list into a single task prompt for the agent."""
    if not messages:
        return ""
    # Simple heuristic: if there's exactly one user message, use its content.
    # Otherwise prepend role labels so the agent sees the full trace.
    if len(messages) == 1 and messages[0].get("role") == "user":
        return str(messages[0].get("content", ""))
    parts = []
    for m in messages:
        role = m.get("role", "user")
        content = m.get("content", "")
        parts.append(f"[{role}] {content}")
    return "\n\n".join(parts)


def _normalize_tools(tools: list[str] | None) -> list[str]:
    if not tools:
        return ["terminal", "file_editor"]
    out = []
    for t in tools:
        tl = str(t).lower()
        # Back-compat: baseline_agent_code.py passes tools=["python"]; map to terminal.
        if tl in ("python", "py", "code", "terminal", "bash", "shell"):
            if "terminal" not in out:
                out.append("terminal")
        elif tl in ("file_editor", "file", "editor", "fs"):
            if "file_editor" not in out:
                out.append("file_editor")
        # Any browser/search/MCP request is silently dropped per HARD RULE.
    return out or ["terminal", "file_editor"]


_sandbox_reach: dict[str, bool] = {}
_sandbox_reach_lock = threading.Lock()


def _sandbox_reaches(base_url: str) -> bool:
    """Whether a process inside the network sandbox can still reach the model
    server. A new network namespace has no interfaces, so today this is false
    for every server; it is probed rather than assumed so that a sandbox which
    leaves a route open is used automatically. Probed once per endpoint."""
    with _sandbox_reach_lock:
        if base_url not in _sandbox_reach:
            from urllib.parse import urlparse
            url = urlparse(base_url)
            host = url.hostname or "localhost"
            port = url.port or (443 if url.scheme == "https" else 80)
            probe = ("import socket, sys\n"
                     "try:\n"
                     "    socket.create_connection((sys.argv[1], int(sys.argv[2])), 3).close()\n"
                     "except Exception:\n"
                     "    sys.exit(1)\n")
            try:
                done = subprocess.run(
                    ["unshare", "--net", "--map-root-user", sys.executable, "-c", probe, host, str(port)],
                    capture_output=True, timeout=20)
                ok = done.returncode == 0
            except Exception:
                ok = False
            _sandbox_reach[base_url] = ok
            if not ok:
                print(f"[agent] the network sandbox (unshare --net) cuts the agent off from the model "
                      f"server {base_url}; agent subprocesses run without it. "
                      f"Set STRATEGIST_AGENT_NO_SANDBOX=0 to force the sandbox.", flush=True)
    return _sandbox_reach[base_url]


def _build_command(use_sandbox: bool) -> list[str]:
    cmd: list[str] = []
    if use_sandbox and shutil.which("unshare"):
        # Network isolation: new netns with no interfaces => no outbound.
        # The orchestrator->vLLM call is made OUTSIDE this subprocess (by the
        # runner's litellm client), which is inside the same netns. This may
        # break vLLM reachability on some clusters — if so, set
        # STRATEGIST_AGENT_NO_SANDBOX=1 to disable.
        cmd.extend(["unshare", "--net", "--map-root-user"])
    cmd.extend([_OPENHANDS_PYTHON, _RUNNER_PATH])
    return cmd


async def _run_via_worker(payload: dict, timeout: float, worker_url: str) -> dict:
    """Forward the payload to a remote openhands_worker HTTP service instead
    of spawning a local subprocess. Keeps openhands subprocess spawns off the
    login node. On worker-unreachable we
    raise so callers don't silently fall back to local behavior — that would
    re-introduce the GPFS metadata storm on vis2."""
    import aiohttp
    async with _get_subproc_sem():
        async with aiohttp.ClientSession() as session:
            try:
                async with session.post(
                    worker_url.rstrip("/") + "/run",
                    json={"payload": payload, "timeout": timeout},
                    timeout=aiohttp.ClientTimeout(total=timeout + 60),
                ) as resp:
                    if resp.status != 200:
                        body = await resp.text()
                        return {"final_text": "", "error": f"worker {resp.status}: {body[:500]}"}
                    return await resp.json()
            except asyncio.TimeoutError:
                return {"final_text": "", "error": f"worker http timed out after {timeout}s"}
            except Exception as e:
                return {"final_text": "", "error": f"worker http error: {e}"}


async def _run_subprocess(payload: dict, timeout: float) -> dict:
    """Run the OpenHands runner once, return the parsed JSON result.

    If STRATEGIST_OPENHANDS_WORKER_URL is set, forward the call over HTTP to
    a remote openhands_worker on a compute node. Otherwise spawn locally."""
    worker_url = os.environ.get("STRATEGIST_OPENHANDS_WORKER_URL", "").strip()
    if worker_url:
        return await _run_via_worker(payload, timeout, worker_url)

    env = dict(os.environ)
    # Make sure the child can import `strategist` if it ever needs to.
    env["PYTHONPATH"] = _REPO_ROOT + os.pathsep + env.get("PYTHONPATH", "")
    env.setdefault("OPENHANDS_SUPPRESS_BANNER", "1")

    flag = env.get("STRATEGIST_AGENT_NO_SANDBOX", "").strip()
    if flag:
        use_sandbox = not bool(int(flag))
    else:
        # Not set: sandbox only if the model server stays reachable from inside it.
        use_sandbox = bool(shutil.which("unshare")) and await asyncio.to_thread(
            _sandbox_reaches, str(payload.get("base_url") or ""))
    cmd = _build_command(use_sandbox)

    # Global cap on concurrent OpenHands subprocesses — see _AGENT_MAX_SUBPROCS.
    async with _get_subproc_sem():
        # start_new_session=True puts the child in its own process group so we
        # can kill the entire group (parent + any grandchildren spawned by
        # openhands-sdk's litellm async workers). Prior incident: SIGKILL on
        # just the parent left hundreds of orphan grandchildren holding RAM.
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
            start_new_session=True,
        )
        pgid = proc.pid  # new session leader's pid IS the pgid
        timed_out = False
        try:
            stdout_b, stderr_b = await asyncio.wait_for(
                proc.communicate(input=json.dumps(payload).encode("utf-8")),
                timeout=timeout,
            )
        except asyncio.TimeoutError:
            timed_out = True
            stdout_b = b""
            stderr_b = b""
        finally:
            # Always group-kill as a defensive cleanup: on timeout the parent
            # is still running; on normal exit, lingering grandchildren
            # (litellm threads, httpx clients) may still hold file handles.
            # ProcessLookupError on normal path means the group is already
            # empty — not a problem.
            try:
                os.killpg(pgid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            except Exception:
                pass
            try:
                await asyncio.wait_for(proc.wait(), timeout=5.0)
            except (asyncio.TimeoutError, ProcessLookupError):
                pass
        if timed_out:
            return {
                "final_text": "",
                "error": f"OpenHands subprocess timed out after {timeout}s",
            }

    stdout = stdout_b.decode("utf-8", errors="replace").strip()
    stderr = stderr_b.decode("utf-8", errors="replace")

    # Runner emits exactly one JSON line on stdout; find it (robust to stray prints).
    parsed: dict | None = None
    for line in reversed(stdout.splitlines()):
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            try:
                parsed = json.loads(line)
                break
            except json.JSONDecodeError:
                continue
    if parsed is None:
        return {
            "final_text": "",
            "error": f"OpenHands runner produced no JSON output. stderr:\n{stderr[-2000:]}",
        }
    # If parsed has empty/short final_text and there's interesting stderr, attach it.
    if (not parsed.get("final_text") or len(str(parsed.get("final_text"))) < 5) and stderr:
        try:
            import time as _t
            dbg = Path("/tmp") / f"agent_stderr_{int(_t.time()*1000)}_{os.getpid()}.log"
            dbg.write_text(f"=== final_text={parsed.get('final_text')!r} ===\n=== STDOUT ===\n{stdout[-3000:]}\n=== STDERR ===\n{stderr[-3000:]}")
        except Exception:
            pass
    return parsed


async def agent(
    llm=None,  # kept for compatibility; IGNORED (we build our own LLM from endpoint)
    messages: list[dict] | None = None,
    tools: list[str] | None = None,
    system: str = "",
    reasoning: bool = False,
    temperature: float = 0.0,
    max_tokens: int = 8192,
    max_steps: int = 10,
) -> str:
    """Run an OpenHands-powered agent on the given messages and return its final text.

    The ``llm`` argument is accepted for backward compatibility with callers
    that pass the shared AsyncOpenAI client; it is ignored. The agent is
    built fresh inside the subprocess against the endpoint configured via
    strategist.basic.generate.set_backend().
    """
    if max_tokens < 2048:
        # HARD RULE: Qwen3-Thinking needs generous max_tokens.
        max_tokens = 8192

    base_url, model, enable_thinking = _resolve_base_url_and_model(llm)
    # reasoning=True forces thinking mode ON even if the global toggle is off.
    if reasoning:
        enable_thinking = True

    normalized_tools = _normalize_tools(tools)
    # normalized_tools is currently informational only; the runner always
    # loads Terminal + FileEditor and they're gated below anyway. Pass along
    # so the runner could filter if we extend it later.

    task = _messages_to_task(messages or [])
    payload = {
        "task": task,
        "system": system or "",
        "base_url": base_url,
        "model": model,
        "enable_thinking": enable_thinking,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "max_steps": max_steps,
        "tools": normalized_tools,
        # vLLM pods are launched with --enable-auto-tool-choice --tool-call-parser hermes
        # so native tool calling is ON by default. Override with STRATEGIST_AGENT_NATIVE_TOOLS=0.
        "native_tool_calling": bool(
            int(os.environ.get("STRATEGIST_AGENT_NATIVE_TOOLS", "1") or "0")
        ),
    }

    # Rough per-call timeout: max_steps * 5 minutes + 60s overhead.
    timeout = max(600.0, float(max_steps) * 300.0 + 60.0)

    import time as _time
    _t0 = _time.perf_counter()
    result = await _run_subprocess(payload, timeout=timeout)
    _elapsed = _time.perf_counter() - _t0

    if not isinstance(result, dict):
        return f"[agent error] subprocess returned {type(result).__name__}, not dict"

    try:
        from strategist.search import _cost_var
        records = _cost_var.get()
        if records is not None:
            tier = getattr(llm, "_bp_tier", None) or "complex"
            mode = getattr(llm, "_bp_mode", None) or ("thinking" if enable_thinking else "nothink")
            records.append({
                "tier": tier,
                "mode": mode,
                "in_tok": int(result.get("prompt_tokens", 0) or 0),
                "out_tok": int(result.get("completion_tokens", 0) or 0),
                "latency": round(float(_elapsed), 2),
                "source": "agent",
                # the whole run, so the generations of an agent call are kept too
                "endpoint": base_url,
                "model": model,
                "enable_thinking": enable_thinking,
                "request": {"task": task, "system": system or "", "tools": normalized_tools,
                            "max_tokens": max_tokens, "max_steps": max_steps, "temperature": temperature},
                "response": {"final_text": str(result.get("final_text", "")),
                             "num_llm_calls": int(result.get("num_llm_calls", 0) or 0),
                             "error": result.get("error")},
                "transcript": result.get("transcript"),
            })
    except Exception:
        pass

    if result.get("error"):
        return f"[agent error] {result['error']}"
    return str(result.get("final_text", "")).strip()
