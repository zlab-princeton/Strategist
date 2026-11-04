from strategist.basic import generate

META = {
    "name": "review",
    "description": "Evaluate content against given criteria. Returns whether it passed and detailed feedback.",
    "when_to_use": "When you need to judge quality, correctness, or any criteria before deciding next steps.",
    "inputs": {
        "content": {
            "type": "str",
            "required": True,
            "description": "The content to review"
        },
        "criteria": {
            "type": "str",
            "required": False,
            "default": "correctness",
            "description": "What to evaluate against. Customizable per session."
        }
    },
    "output": "(passed: bool, feedback: str)",
    "tags": ["evaluation", "quality"],
    "uses": ["generate"],
}

PROMPT = """Evaluate the following content against this criteria: {criteria}

If the content is acceptable, respond with exactly "PASS" on the first line, followed by a brief explanation.
If the content has issues, respond with "FAIL" on the first line, followed by a list of specific problems.

Content:
{content}"""


async def review(llm, content: str, criteria: str = "correctness") -> tuple[bool, str]:
    response = await generate(llm, PROMPT.format(criteria=criteria, content=content))
    passed = response.strip().startswith("PASS")
    return passed, response
