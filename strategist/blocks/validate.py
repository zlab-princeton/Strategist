from strategist.basic import generate

META = {
    "name": "validate",
    "description": "Check whether output satisfies a specific correctness condition.",
    "when_to_use": "When you need a binary yes/no check on whether an output meets a condition.",
    "inputs": {
        "output": {
            "type": "str",
            "required": True,
            "description": "The output to validate"
        },
        "condition": {
            "type": "str",
            "required": True,
            "description": "The condition that must be true"
        }
    },
    "output": "(valid: bool, explanation: str)",
    "tags": ["validation", "correctness"],
    "uses": ["generate"],
}

PROMPT = """Determine whether the following output satisfies this condition: {condition}

Output:
{output}

Respond with "VALID" or "INVALID" on the first line, followed by a brief explanation."""


async def validate(llm, output: str, condition: str) -> tuple[bool, str]:
    response = await generate(llm, PROMPT.format(condition=condition, output=output))
    valid = response.strip().startswith("VALID")
    return valid, response
