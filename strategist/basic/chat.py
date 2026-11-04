import openai

META = {
    "name": "chat",
    "description": "Multi-turn LLM call. Send conversation history, get the next response.",
    "inputs": {
        "messages": {
            "type": "list[dict]",
            "required": True,
            "description": "Conversation history. Each message: {'role': 'user'|'assistant', 'content': '...'}"
        },
        "system": {
            "type": "str",
            "required": False,
            "default": "",
            "description": "System instruction"
        },
        "reasoning": {
            "type": "bool",
            "required": False,
            "default": False,
            "description": "Enable extended reasoning/thinking mode."
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
        }
    },
    "output": "str — the model's next response",
    "tags": ["basic", "multi-turn"],
}

# Uses the same backend as generate — imported lazily inside chat()


async def chat(
    llm=None,
    messages: list[dict] = None,
    system: str = "",
    reasoning: bool = False,
    temperature: float = 0.0,
    max_tokens: int = 16384
) -> str:
    from strategist.basic.generate import _client, _model

    client = llm or _client
    if client is None:
        raise RuntimeError("No backend configured. Call generate.set_backend() or pass llm.")

    full_messages = []
    if system:
        full_messages.append({"role": "system", "content": system})

    if reasoning and messages:
        # Append reasoning instruction to the last user message
        msgs_copy = [dict(m) for m in messages]
        for i in range(len(msgs_copy) - 1, -1, -1):
            if msgs_copy[i]["role"] == "user":
                msgs_copy[i]["content"] += "\n\nThink step by step, showing your reasoning."
                break
        full_messages.extend(msgs_copy)
    else:
        full_messages.extend(messages or [])

    from strategist.basic.generate import _enable_thinking, strip_thinking, _clamp_max_tokens
    client_model = getattr(client, "_bp_model_name", None) or _model
    client_thinking = getattr(client, "_bp_enable_thinking", _enable_thinking)
    provider = getattr(client, "_bp_provider", "vllm")
    if provider == "vllm":
        effective_max = _clamp_max_tokens(max_tokens, full_messages)
        kwargs = {
            "model": client_model,
            "messages": full_messages,
            "temperature": temperature,
            "max_tokens": effective_max,
            "extra_body": {"chat_template_kwargs": {"enable_thinking": client_thinking}},
        }
    else:
        kwargs = {"model": client_model, "messages": full_messages}
        if provider == "openai":
            kwargs["max_completion_tokens"] = max_tokens
        else:
            kwargs["max_tokens"] = max_tokens
        effort = getattr(client, "_bp_reasoning_effort", None)
        if effort:
            kwargs["reasoning_effort"] = effort
    response = await client.chat.completions.create(**kwargs)
    return strip_thinking(response.choices[0].message.content or "")
