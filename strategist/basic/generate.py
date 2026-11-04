import os
import re

import openai

# Qwen3-Thinking models wrap reasoning in <think>...</think> tags.
# Without --enable-reasoning in vLLM, these tags appear in message.content.
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


def strip_thinking(text: str) -> str:
    """Remove <think>...</think> blocks from model output.

    With complete tags: removes the block, keeps content after.
    With truncated <think> (no </think>): strips the tag but keeps the text,
    since the model's reasoning often contains the answer and downstream
    eval can extract it.
    """
    if not text:
        return text

    # Case 1: complete <think>...</think> — remove the block, keep rest
    cleaned = _THINK_RE.sub("", text)

    # Case 2: stray </think> without opening tag (template injected <think>)
    if "</think>" in cleaned:
        cleaned = cleaned.split("</think>", 1)[-1]

    # Case 3: <think> without </think> — truncated by max_tokens.
    # Just remove the tag itself, keep the content — the reasoning text
    # often contains the answer and eval code can extract it.
    cleaned = cleaned.replace("<think>", "")

    return cleaned.strip()


META = {
    "name": "generate",
    "description": "Single LLM call. Give a prompt, get a response.",
    "inputs": {
        "prompt": {
            "type": "str",
            "required": True,
            "description": "The user-facing prompt to send to the model"
        },
        "system": {
            "type": "str",
            "required": False,
            "default": "",
            "description": "System/role instruction prepended to the call"
        },
        "reasoning": {
            "type": "bool",
            "required": False,
            "default": False,
            "description": "Enable extended reasoning/thinking mode. Slower but better for complex tasks."
        },
        "temperature": {
            "type": "float",
            "required": False,
            "default": 0.0,
            "description": "Sampling temperature"
        },
        "max_tokens": {
            "type": "int",
            "required": False,
            "default": 16384,
            "description": "Maximum response length"
        },
        "stop": {
            "type": "list[str] | None",
            "required": False,
            "default": None,
            "description": "Stop sequences"
        }
    },
    "output": "str — the model's response",
    "tags": ["basic", "single-turn"],
}

# Default vLLM endpoint — override via set_backend()
_client: openai.AsyncOpenAI | None = None
_model: str = "Qwen/Qwen3-4B"
# Global thinking-mode toggle for hybrid Qwen3 (0.6B/1.7B/4B). No-op for Thinking-2507.
_enable_thinking: bool = True


def set_backend(base_url: str = "http://localhost:8000/v1", model: str = "Qwen/Qwen3-4B", api_key: str = "EMPTY"):
    """Configure the vLLM backend. Call once at startup."""
    global _client, _model
    # timeout=3600: complex/thinking calls with max_tokens=8192 under fleet-wide
    # vLLM queue contention can take 15-30+ min on hard benchmarks (AIME, math500).
    # Default 600s kills them mid-response, producing APITimeoutError on the bulk
    # of samples. SLURM walltime is the real outer bound.
    _client = openai.AsyncOpenAI(base_url=base_url, api_key=api_key, timeout=3600.0)
    _model = model


def set_thinking(on: bool) -> None:
    """Toggle hybrid-Qwen3 thinking mode on/off for all subsequent generate/chat calls."""
    global _enable_thinking
    _enable_thinking = on


# Hard cap on requested output tokens. Every strategy LLM call uses
# max_tokens=16384 (paper A.4); the clamp below only shrinks the request when
# the serving window is too small (input_tokens + max_tokens > max_model_len
# → 400 from vLLM). The per-deployment context window is configurable via
# STRATEGIST_MODEL_CTX (default 32768 — the Qwen3 native window).
_MAX_OUT_TOKENS = 16384
_DEFAULT_MODEL_CTX = int(os.environ.get("STRATEGIST_MODEL_CTX", "32768"))
# Use the real Qwen3 tokenizer when available; falls back to a conservative
# char-based estimate if load fails. Char estimate was undercounting real
# tokens (observed: input=2352 when char-est said ~1500), causing 300+ overflow
# errors under long-context baselines.
_SAFETY_MARGIN = 500
_TOKENIZER = None


def _get_tokenizer():
    global _TOKENIZER
    if _TOKENIZER is False:
        return None
    if _TOKENIZER is None:
        try:
            from transformers import AutoTokenizer
            _TOKENIZER = AutoTokenizer.from_pretrained(
                os.environ.get("STRATEGIST_TOKENIZER", "Qwen/Qwen3-4B"),
                trust_remote_code=True,
            )
        except Exception:
            _TOKENIZER = False
            return None
    return _TOKENIZER


def _estimate_input_tokens(messages: list[dict]) -> int:
    tok = _get_tokenizer()
    if tok is not None:
        try:
            # Token count including role/special tokens. Depending on the
            # transformers version this returns a list of ids or a
            # BatchEncoding ({"input_ids": [...]}) — len() of the latter is the
            # number of keys (2), which silently disabled the clamp and let
            # long prompts hit the server's window (400 errors).
            out = tok.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
            if hasattr(out, "keys"):
                out = out["input_ids"]
            if out and isinstance(out[0], (list, tuple)):
                out = out[0]
            n = len(out)
            if n >= 4:
                return n
        except Exception:
            pass
    # Fallback: conservative char/token ratio (2.0 over-estimates, safer).
    total_chars = sum(len(str(m.get("content", ""))) for m in messages)
    return int(total_chars / 2.0) + len(messages) * 16


def _clamp_max_tokens(requested: int, messages: list[dict], model_ctx: int | None = None) -> int:
    if model_ctx is None:
        model_ctx = _DEFAULT_MODEL_CTX
    est_input = _estimate_input_tokens(messages)
    budget = model_ctx - est_input - _SAFETY_MARGIN
    return max(64, min(requested, _MAX_OUT_TOKENS, budget))


async def generate(
    llm=None,
    prompt: str = "",
    system: str = "",
    reasoning: bool = False,
    temperature: float = 0.0,
    max_tokens: int = 16384,
    stop: list[str] | None = None
) -> str:
    client = llm or _client
    if client is None:
        raise RuntimeError("No backend configured. Call generate.set_backend() or pass llm.")

    messages = []
    if system:
        messages.append({"role": "system", "content": system})

    if reasoning:
        prompt = prompt + "\n\nThink step by step, showing your reasoning before giving a final answer."

    messages.append({"role": "user", "content": prompt})

    # Env-gated floor (off by default): library blueprints written before the
    # paper A.4 rule carry small explicit caps (640/8192) that truncate
    # thinking-mode output. Ablation harnesses set STRATEGIST_MIN_MAXTOK=16384
    # so every arm runs under the same cap without editing measured artifacts.
    _floor = os.environ.get("STRATEGIST_MIN_MAXTOK")
    if _floor:
        max_tokens = max(max_tokens, int(_floor))

    # Per-client overrides (multi-tier): when a traced client
    # carries `_bp_model_name` and/or `_bp_enable_thinking`, use those instead
    # of the module-level defaults. Back-compat: absent attrs → defaults.
    client_model = getattr(client, "_bp_model_name", None) or _model
    client_thinking = getattr(client, "_bp_enable_thinking", _enable_thinking)
    # Cross-model transfer (paper A.7): clients tagged with a provider other
    # than the local vLLM stack drop the vLLM-specific knobs and carry the
    # provider's own reasoning control instead. Sampling params are omitted
    # because API reasoning models reject non-default values.
    provider = getattr(client, "_bp_provider", "vllm")
    if provider == "vllm":
        effective_max = _clamp_max_tokens(max_tokens, messages)
        kwargs = {
            "model": client_model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": effective_max,
            # Enable thinking mode for hybrid Qwen3 models (0.6B/1.7B/4B/8B).
            # No-op for Thinking-2507 (already thinking-only).
            "extra_body": {"chat_template_kwargs": {"enable_thinking": client_thinking}},
        }
    else:
        kwargs = {"model": client_model, "messages": messages}
        if provider == "openai":
            kwargs["max_completion_tokens"] = max_tokens
        else:
            kwargs["max_tokens"] = max_tokens
        effort = getattr(client, "_bp_reasoning_effort", None)
        if effort:
            kwargs["reasoning_effort"] = effort
    if stop:
        kwargs["stop"] = stop

    response = await client.chat.completions.create(**kwargs)
    msg = response.choices[0].message

    # With --reasoning-parser deepseek_r1: reasoning_content holds the thinking,
    # content holds the final answer (may be empty). Prioritize content.
    if msg.content:
        return strip_thinking(msg.content)
    # Fallback: reasoning_content may contain the answer when content is empty
    # (e.g., model didn't explicitly separate answer from reasoning).
    if hasattr(msg, 'reasoning_content') and msg.reasoning_content:
        return strip_thinking(msg.reasoning_content)
    return ""


# Compat re-export: LLM-authored workflows frequently write
# `from strategist.basic.generate import chat` even though chat lives in
# strategist.basic.chat. Aliasing here makes those imports succeed instead
# of tanking the whole iteration.
from strategist.basic.chat import chat  # noqa: E402,F401
