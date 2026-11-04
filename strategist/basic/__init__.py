from .generate import generate
from .chat import chat
from .agent import agent

# Compat shim: blueprints may call `await llm.generate(...)` method-style instead
# of the canonical `await generate(llm, ...)`. Both styles are accepted: `.generate`
# and `.agent` are attached to openai.AsyncOpenAI here, and the search loop's
# `_ClientView` gets the same methods at construction time. `chat` is not added:
# openai clients already expose `.chat.completions.create`.
import openai as _openai


async def _client_generate(self, *args, **kwargs):
    return await generate(self, *args, **kwargs)


async def _client_agent(self, *args, **kwargs):
    return await agent(self, *args, **kwargs)


if not hasattr(_openai.AsyncOpenAI, "generate"):
    _openai.AsyncOpenAI.generate = _client_generate  # type: ignore[attr-defined]
if not hasattr(_openai.AsyncOpenAI, "agent"):
    _openai.AsyncOpenAI.agent = _client_agent  # type: ignore[attr-defined]
