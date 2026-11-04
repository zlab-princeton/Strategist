"""OpenHands agent subprocess runner (Python 3.12 entrypoint).

This file is invoked by strategist/basic/agent.py as a subprocess using the
Python that has OpenHands installed (the repo-root .venv by default, or
STRATEGIST_OPENHANDS_PYTHON).

Protocol:
  stdin:  JSON payload with keys: task, system, base_url, model,
          enable_thinking, temperature, max_tokens, max_steps, workspace
  stdout: A single JSON line with keys: final_text, num_llm_calls,
          prompt_tokens, completion_tokens, error

Tools: TerminalTool + FileEditorTool ONLY. Never browser/search/MCP.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import traceback

os.environ.setdefault("OPENHANDS_SUPPRESS_BANNER", "1")
os.environ.setdefault("ALLOW_SHORT_CONTEXT_WINDOWS", "true")


def _text(obj, limit: int = 30000):
    """Plain text of an event field: TextContent lists, pydantic models, strings."""
    try:
        if obj is None:
            return None
        if isinstance(obj, str):
            text = obj
        elif isinstance(obj, (list, tuple)):
            parts = []
            for item in obj:
                t = getattr(item, "text", None)
                parts.append(t if isinstance(t, str) else _text(item, limit))
            text = "\n".join(str(part) for part in parts if part)
        elif hasattr(obj, "model_dump"):
            text = json.dumps(obj.model_dump(mode="json"), ensure_ascii=False, default=str)
        else:
            text = str(obj)
        return text if len(text) <= limit else text[:limit] + f"... [{len(text) - limit} more chars]"
    except Exception as exc:  # the transcript is a record, never a reason to fail
        return f"[unserializable: {exc}]"


def _transcript(events) -> list:
    """Every event of the conversation: the model's messages and tool calls, and
    the tool results, so an agent run can be read back in full."""
    out = []
    for ev in events:
        rec = {"type": type(ev).__name__, "source": str(getattr(ev, "source", "") or "")}
        for field in ("llm_message", "thought", "action", "tool_name", "tool_call",
                      "observation", "error", "system_prompt"):
            if hasattr(ev, field):
                value = getattr(ev, field)
                if value is not None:
                    rec[field] = _text(value)
        out.append(rec)
    return out


def main() -> None:
    try:
        payload = json.load(sys.stdin)
    except Exception as e:
        print(json.dumps({"error": f"bad stdin json: {e}"}))
        return

    task = payload.get("task", "")
    system = payload.get("system", "") or ""
    base_url = payload["base_url"]
    model = payload.get("model", "Qwen/Qwen3-4B")
    enable_thinking = bool(payload.get("enable_thinking", True))
    temperature = float(payload.get("temperature", 0.0))
    max_tokens = int(payload.get("max_tokens", 8192))
    max_steps = int(payload.get("max_steps", 10))
    native_tool_calling = bool(payload.get("native_tool_calling", False))
    workspace_path = payload.get("workspace") or None

    try:
        from openhands.sdk import LLM, Agent, LocalConversation, Tool

        # Import tools to register them before creating the agent
        from openhands.tools.terminal.definition import TerminalTool
        from openhands.tools.file_editor.definition import FileEditorTool

        # Trigger tool registration by importing their modules
        # This is required because tools register themselves on import
    except Exception as e:
        print(json.dumps({"error": f"openhands import failed: {e}"}))
        return

    llm = LLM(
        usage_id="blueprint-agent",
        model=f"openai/{model}",
        api_key="EMPTY",
        base_url=base_url,
        # Cap at 8192, NOT an A.4 violation: multi-turn agent context (system
        # + tool schemas + step outputs) plus a 16384 completion would exceed
        # the 32768 serving window and vLLM 400s every call.
        max_output_tokens=min(max_tokens, 8192),
        temperature=temperature,
        native_tool_calling=native_tool_calling,
        litellm_extra_body={"chat_template_kwargs": {"enable_thinking": enable_thinking}},
        timeout=600,
        disable_vision=True,
        num_retries=3,
    )

    # Tools: Terminal + FileEditor ONLY. No browser, no MCP, no web search.
    # Note: Tool names must match the registered names (lowercase: 'terminal', 'file_editor')
    tools = [
        Tool(name="terminal", params={}),
        Tool(name="file_editor", params={}),
    ]

    result = {
        "final_text": "",
        "num_llm_calls": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "error": None,
    }

    tmp_ctx = None
    try:
        if workspace_path is None:
            tmp_ctx = tempfile.TemporaryDirectory(prefix="oh_agent_")
            workspace_path = tmp_ctx.name

        # Create agent with tools
        agent = Agent(llm=llm, tools=tools)

        # Create conversation with workspace
        conv = LocalConversation(
            agent=agent,
            workspace=workspace_path,
            max_iteration_per_run=max_steps,
            stuck_detection=False,  # Disable to prevent premature termination
        )

        # Add the task as a user message
        if system:
            full_task = f"{system}\n\n---\n\n{task}"
        else:
            full_task = task
        conv.send_message(full_task)

        # Run the conversation
        conv.run()

        # Extract final response. The newer openhands API expects a
        # Sequence[Event], not the conversation object — passing `conv`
        # raises `'LocalConversation' object is not reversible`. Materialize
        # the event log into a list which is always reversible.
        from openhands.sdk.conversation.response_utils import get_agent_final_response
        try:
            events = list(conv.state.events)
        except Exception:
            events = []
        final_response = get_agent_final_response(events) if events else ""
        result["final_text"] = final_response or ""
        try:
            result["transcript"] = _transcript(events)
        except Exception:
            result["transcript"] = []

        # Token usage: newer SDK exposes per-LLM Metrics via
        # conv.conversation_stats.get_combined_metrics(). Each Metrics
        # carries accumulated_token_usage (TokenUsage with prompt+completion)
        # and a token_usages list whose length is the call count.
        try:
            stats = getattr(conv, "conversation_stats", None) or getattr(conv, "stats", None)
            if stats is not None and hasattr(stats, "get_combined_metrics"):
                metrics = stats.get_combined_metrics()
                acc = getattr(metrics, "accumulated_token_usage", None)
                if acc is not None:
                    result["prompt_tokens"] = int(getattr(acc, "prompt_tokens", 0) or 0)
                    result["completion_tokens"] = int(getattr(acc, "completion_tokens", 0) or 0)
                # Call count: prefer length of token_usages list if present.
                tu_list = getattr(metrics, "token_usages", None)
                if tu_list is not None:
                    result["num_llm_calls"] = len(tu_list)
        except Exception:
            pass

    except Exception as e:
        result["error"] = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"
    finally:
        if tmp_ctx is not None:
            try:
                tmp_ctx.cleanup()
            except Exception:
                pass

    sys.stdout.write(json.dumps(result) + "\n")
    sys.stdout.flush()


if __name__ == "__main__":
    main()