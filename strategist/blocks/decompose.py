import json
from strategist.basic import generate

META = {
    "name": "decompose",
    "description": "Break a complex task into smaller, independent sub-tasks.",
    "when_to_use": "When a task is too complex to solve in one step. Split it, solve each part, then aggregate.",
    "inputs": {
        "task": {
            "type": "str",
            "required": True,
            "description": "The task to decompose"
        },
        "hint": {
            "type": "str",
            "required": False,
            "default": "",
            "description": "Guidance on how to split. Customizable per session."
        }
    },
    "output": "list[str] — the sub-tasks",
    "tags": ["decomposition", "planning"],
    "uses": ["generate"],
}

PROMPT = """Break the following task into smaller, independent sub-tasks.
{hint_line}
Return the sub-tasks as a JSON list of strings. Example: ["sub-task 1", "sub-task 2", "sub-task 3"]

Task:
{task}

Sub-tasks (JSON list):"""


async def decompose(llm, task: str, hint: str = "") -> list[str]:
    hint_line = f"Hint: {hint}" if hint else ""
    response = await generate(llm, PROMPT.format(hint_line=hint_line, task=task))
    # Parse JSON list from response
    response = response.strip()
    # Find the JSON array in the response
    start = response.find("[")
    end = response.rfind("]") + 1
    if start != -1 and end > start:
        return json.loads(response[start:end])
    # Fallback: split by newlines
    return [line.strip("- ").strip() for line in response.strip().split("\n") if line.strip()]
