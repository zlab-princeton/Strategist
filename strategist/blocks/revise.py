from strategist.basic import generate

META = {
    "name": "revise",
    "description": "Improve content based on specific feedback. Keeps what works, fixes what's broken.",
    "when_to_use": "After a review identifies problems. Pair with the review block.",
    "inputs": {
        "content": {
            "type": "str",
            "required": True,
            "description": "The content to improve"
        },
        "feedback": {
            "type": "str",
            "required": True,
            "description": "Specific feedback on what to fix"
        },
        "instruction": {
            "type": "str",
            "required": False,
            "default": "",
            "description": "Additional revision guidance. Customizable per session."
        }
    },
    "output": "str — the revised content",
    "tags": ["refinement", "improvement"],
    "uses": ["generate"],
}

PROMPT = """Revise the following content to address the feedback below.
Keep everything that works. Only change what needs fixing.
{instruction_line}
Feedback:
{feedback}

Original content:
{content}

Revised content:"""


async def revise(llm, content: str, feedback: str, instruction: str = "") -> str:
    instruction_line = f"\nAdditional instruction: {instruction}" if instruction else ""
    response = await generate(llm, PROMPT.format(
        instruction_line=instruction_line,
        feedback=feedback,
        content=content
    ))
    return response
