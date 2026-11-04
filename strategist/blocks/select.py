from strategist.basic import generate

META = {
    "name": "select",
    "description": "Pick the best option from a list of candidates based on given criteria.",
    "when_to_use": "When you have multiple candidate outputs and need to choose the best one.",
    "inputs": {
        "candidates": {
            "type": "list[str]",
            "required": True,
            "description": "The candidate options to choose from"
        },
        "criteria": {
            "type": "str",
            "required": False,
            "default": "overall quality",
            "description": "What 'best' means. Customizable per session."
        }
    },
    "output": "str — the selected candidate",
    "tags": ["selection", "ensemble"],
    "uses": ["generate"],
}

PROMPT = """Below are {n} candidate responses. Select the best one based on: {criteria}

{candidates_text}

Reply with the number of the best candidate (e.g. "1"), followed by a brief justification."""


async def select(llm, candidates: list[str], criteria: str = "overall quality") -> str:
    candidates_text = "\n\n".join(
        f"--- Candidate {i+1} ---\n{c}" for i, c in enumerate(candidates)
    )
    response = await generate(llm, PROMPT.format(
        n=len(candidates),
        criteria=criteria,
        candidates_text=candidates_text
    ))
    # Parse the selected number and return that candidate
    for i, c in enumerate(candidates):
        if response.strip().startswith(str(i + 1)):
            return c
    # Fallback: return first candidate if parsing fails
    return candidates[0]
