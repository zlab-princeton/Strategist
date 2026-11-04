"""The strategist LLM backends: a prompt in, the strategist's reply out.

    codex           the Codex CLI (`codex exec`), one resumable session per run;
                    model from STRATEGIST_MODEL
    claude-opus     the Anthropic API (ANTHROPIC_API_KEY)
    claude-minimax  the Claude Code CLI on a MiniMax model through an
                    Anthropic-compatible endpoint (MINIMAX_API_KEY)
"""
from __future__ import annotations

import json
import os
import random
import subprocess
import tempfile
import time
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent


class CodexSessionLost(RuntimeError):
    """Resuming the run's Codex session failed 3 times; its id file was removed,
    so the caller starts a new session with the full prompt."""


def codex_call(prompt: str, timeout_sec: float = 1800.0,
                    session_file: "Path | None" = None) -> str:
    """Strategist via the Codex CLI (ChatGPT subscription auth; no API key).

    Model comes from ~/.codex/config.toml (currently gpt-5.6-sol, reasoning
    effort high); override with STRATEGIST_MODEL.
      - runs in an empty temp dir with --skip-git-repo-check so there is no
        project context to explore;
      - read-only sandbox plus an explicit no-tools instruction;
      - the reply is read from --output-last-message (clean final answer),
        never scraped from the progress stream on stdout.

    Without `session_file` every call is a new session that sees only its
    prompt (the paper's stateless strategist). With it (the search loop's default
    since 2026-10-09; `--no-resume-session` turns it off) the run keeps one
    session: the first call starts it and writes its id (the `thread.started`
    event of `--json`) to the file, later calls resume it (`codex exec resume
    <id>`), so the strategist has every earlier prompt and answer of the run.
    If resuming fails 3 times the file is removed and CodexSessionLost raised.
    """
    import os
    import shutil
    import subprocess
    import tempfile

    import fcntl
    import random

    sid = ""
    if session_file is not None and Path(session_file).exists():
        sid = Path(session_file).read_text().strip()
    guard = (
        "Answer directly in a single message. Do not run any commands, do not "
        "read or write any files; everything you need is in this prompt"
        + (" and earlier in this conversation" if sid else "") + ".\n\n"
    )
    model = os.environ.get("STRATEGIST_MODEL", "gpt-5.6-sol").strip()
    # Reasoning effort: ~/.codex/config.toml unless STRATEGIST_REASONING_EFFORT is set
    # (e.g. "medium"; the one-shot retrieval runs of 2026-10-14 use it).
    effort = os.environ.get("STRATEGIST_REASONING_EFFORT", "").strip()
    effort_args = ["-c", f'model_reasoning_effort="{effort}"'] if effort else []

    def _acquire_slot(max_slots: int = 6, wait_s: float = 900.0):
        """Global cross-process cap on concurrent codex calls: burst
        throttling on the ChatGPT account returns empty rc=1 outputs. Try N
        lock files until one is free."""
        lock_dir = Path.home() / ".cache" / "codex_strategist_locks"
        lock_dir.mkdir(parents=True, exist_ok=True)
        deadline = time.time() + wait_s
        while True:
            for i in range(max_slots):
                fh = open(lock_dir / f"slot{i}.lock", "w")
                try:
                    fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    return fh
                except OSError:
                    fh.close()
            if time.time() > deadline:
                return None
            time.sleep(5 + random.uniform(0, 5))

    last_err = "unknown"
    for attempt in range(3):
        workdir = tempfile.mkdtemp(prefix="codex_strategist_")
        out_path = os.path.join(workdir, "last_message.txt")
        if sid:
            cmd = ["codex", "exec", "resume", "--json", "-c", 'sandbox_mode="read-only"',
                   "--skip-git-repo-check", "--output-last-message", out_path,
                   "-m", model, *effort_args, sid, "-"]
        else:
            cmd = ["codex", "exec", "--sandbox", "read-only",
                   "--skip-git-repo-check", "--output-last-message", out_path,
                   "-m", model, *effort_args, "-"]
            if session_file is not None:
                cmd.insert(2, "--json")
        slot = _acquire_slot()
        try:
            result = subprocess.run(
                cmd,
                input=guard + prompt,
                capture_output=True,
                text=True,
                timeout=timeout_sec,
                cwd=workdir,
            )
            output = ""
            if os.path.exists(out_path):
                with open(out_path) as f:
                    output = f.read().strip()
            if output:
                if session_file is not None and not sid:
                    new_sid = ""
                    for line in (result.stdout or "").splitlines():
                        try:
                            ev = json.loads(line)
                        except ValueError:
                            continue
                        if ev.get("type") == "thread.started" and ev.get("thread_id"):
                            new_sid = ev["thread_id"]
                            break
                    if new_sid:
                        Path(session_file).write_text(new_sid + "\n")
                    else:
                        print("  [codex] no session id in the --json events; the next call starts a new session")
                return output
            err = (result.stderr or "").strip()
            last_err = (f"codex strategist returned empty output "
                        f"(rc={result.returncode}{', resume ' + sid if sid else ''}). stderr: {err[:500]}")
            if "usage limit" in err.lower():
                raise RuntimeError(last_err)  # hard cap: no point retrying
        finally:
            if slot is not None:
                slot.close()
            shutil.rmtree(workdir, ignore_errors=True)
        # Burst throttle: back off before the next attempt.
        time.sleep(45 * (attempt + 1) + random.uniform(5, 30))
    if sid:
        Path(session_file).unlink(missing_ok=True)
        raise CodexSessionLost(last_err)
    raise RuntimeError(last_err)


def claude_opus_call(
    prompt: str,
    model: str | None = None,
    max_tokens: int = 32000,
    timeout_sec: float = 3600.0,
) -> str:
    """Call the strategist directly through the Anthropic API (paper Sec. 4:
    'The strategist is Claude Opus 4.7, a separate frozen LLM accessed via
    API'). Stateless per iteration — continuity across iterations comes from
    the prior-iteration log embedded in the prompt, exactly as in the paper's
    single-prompt-per-iteration template (A.5).

    Auth resolves from the environment (ANTHROPIC_API_KEY / ANTHROPIC_AUTH_TOKEN
    or an `ant auth login` profile). Model override via STRATEGIST_MODEL.
    Streaming keeps the long candidate-slate responses inside HTTP timeouts.
    """
    import os

    import anthropic

    model = model or os.environ.get("STRATEGIST_MODEL", "claude-opus-4-7")
    client = anthropic.Anthropic(timeout=timeout_sec, max_retries=3)
    # The combined system+user prompt arrives as one string (the loop joins
    # them with a --- separator); send it as the user turn.
    with client.messages.stream(
        model=model,
        max_tokens=max_tokens,
        thinking={"type": "adaptive"},
        messages=[{"role": "user", "content": prompt}],
    ) as stream:
        response = stream.get_final_message()

    if response.stop_reason == "refusal":
        raise RuntimeError(f"strategist ({model}) refused the request")
    text = "".join(
        block.text for block in response.content if getattr(block, "type", "") == "text"
    ).strip()
    if not text:
        raise RuntimeError(
            f"strategist ({model}) returned no text (stop_reason={response.stop_reason})"
        )
    return text


def _load_minimax_token() -> str:
    """Token priority: MINIMAX_API_KEY env var, then ~/.config/minimax/token."""
    import os
    env_tok = os.environ.get("MINIMAX_API_KEY", "").strip()
    if env_tok:
        return env_tok
    path = Path.home() / ".config" / "minimax" / "token"
    if path.exists():
        return path.read_text().strip()
    raise RuntimeError(
        "MiniMax token not found. Set MINIMAX_API_KEY env var or write the "
        "token to ~/.config/minimax/token."
    )


def claude_minimax_call(
    prompt: str,
    model: str = "MiniMax-M2.7-highspeed",
    effort: str = "high",
    max_turns: int = 200,
    timeout_sec: int = 1800,
    session_file: "Path | None" = None,
) -> str:
    """Call Claude Code CLI wired to MiniMax via the Anthropic-compatible
    endpoint. Returns the final assistant text.

    Session resumption:
      If `session_file` is given and already has a session ID written to
      it, we pass `--resume <id>` so the commander sees all prior turns
      from this (benchmark, run_seed) — its own previous proposals,
      slate metadata, self-written experience notes, etc. The commander
      is otherwise a fresh subprocess per iteration with total amnesia;
      resumption is the only way a chain of reasoning across iterations
      holds together.

      After the call returns, we parse the stream-json `system` init
      event for the session_id and write it back to `session_file` so
      the next iteration can resume.

    Uses `--print --output-format stream-json`: every stdout line is a JSON
    event. We extract the assistant's text payload from the terminal
    `result` event (or fall back to concatenated assistant text deltas).
    """
    import os
    import subprocess
    from pathlib import Path as _Path

    env = dict(os.environ)
    env["ANTHROPIC_BASE_URL"] = "https://api.minimax.io/anthropic"
    env["ANTHROPIC_AUTH_TOKEN"] = _load_minimax_token()
    env["ANTHROPIC_MODEL"] = model
    env["CLAUDE_CODE_EFFORT_LEVEL"] = effort

    # Isolate Claude Code's state tree inside the repo. The CLI reads
    # `$HOME/.claude/` for projects/ (session history), settings, plugins,
    # etc. Default HOME points at ~/.claude which is shared with the user's
    # own Cursor/Claude Code sessions — not great for a long-running batch
    # of subprocess commanders. Pointing HOME at a repo-local fake home
    # keeps session files under version-controllable scope without
    # touching the user's own state. Auth still works because we inject
    # ANTHROPIC_AUTH_TOKEN directly (no ~/.claude.json read needed).
    from pathlib import Path as _Path
    _claude_home = ROOT_DIR / ".claude_sessions" / "home"
    (_claude_home / ".claude").mkdir(parents=True, exist_ok=True)
    env["HOME"] = str(_claude_home)

    # Pipe prompt via stdin, not argv, to avoid [Errno 7] Argument list too
    # long on benchmarks with long passages (quality, 2wikimqa, hotpotqa,
    # etc.). Linux argv+env cap is ~128 KB; prompts with failure traces
    # from long-context benchmarks overflow it.
    cmd = [
        "claude",
        "--print",
        "--verbose",
        "--output-format", "stream-json",
        "--include-partial-messages",
        "--model", "opus",
        "--max-turns", str(max_turns),
        "--allowedTools", "Bash,Read,Write,Edit,Glob,Grep,WebSearch,WebFetch",
    ]

    # Resume the same local session across iterations for this (bench, run).
    prior_session_id = None
    if session_file is not None:
        try:
            sf = _Path(session_file)
            if sf.exists():
                prior_session_id = sf.read_text().strip() or None
        except Exception:
            prior_session_id = None
    if prior_session_id:
        cmd += ["--resume", prior_session_id]

    proc = subprocess.run(
        cmd,
        input=prompt,
        capture_output=True,
        text=True,
        timeout=timeout_sec,
        env=env,
        cwd=str(ROOT_DIR),
    )

    if proc.returncode != 0 and not proc.stdout.strip():
        raise RuntimeError(
            f"Claude/MiniMax call failed (rc={proc.returncode}). "
            f"stderr: {proc.stderr[:800]}"
        )

    # Parse NDJSON stream. Prefer the terminal `result` event; if absent,
    # concatenate assistant-text content from message events. Also capture
    # the session_id from the first `system` event so we can resume next
    # iteration.
    final_text = ""
    assistant_deltas: list[str] = []
    session_id = None
    for line in proc.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            ev = json.loads(line)
        except json.JSONDecodeError:
            continue
        ev_type = ev.get("type")
        if session_id is None:
            sid = ev.get("session_id") or (ev.get("message") or {}).get("session_id")
            if sid:
                session_id = sid
        if ev_type == "result":
            # Claude Code emits the final text as `result` or under .result.
            final_text = ev.get("result") or ev.get("text") or final_text
        elif ev_type == "assistant":
            # Assistant messages nest content blocks.
            msg = ev.get("message") or {}
            for block in msg.get("content") or []:
                if isinstance(block, dict) and block.get("type") == "text":
                    assistant_deltas.append(block.get("text", ""))

    # Persist the session_id so the next iteration resumes here.
    if session_file is not None and session_id:
        try:
            sf = _Path(session_file)
            sf.parent.mkdir(parents=True, exist_ok=True)
            sf.write_text(session_id)
        except Exception as e:
            print(f"  [session] warning: could not write session_id: {e}")

    output = (final_text or "\n".join(assistant_deltas)).strip()
    if not output:
        raise RuntimeError(
            f"Claude/MiniMax returned no assistant text. stderr: {proc.stderr[:500]}"
        )
    return output
