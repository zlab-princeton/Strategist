from strategist.basic import generate

META = {
    "name": "discuss",
    "description": "Generate multiple viewpoints on a task, then synthesize into a consensus.",
    "when_to_use": "When a task benefits from multiple perspectives before committing to an answer.",
    "inputs": {
        "task": {
            "type": "str",
            "required": True,
            "description": "The topic to discuss"
        },
        "perspectives": {
            "type": "list[str]",
            "required": False,
            "default": ["analytical", "critical", "creative"],
            "description": "The viewpoints to include. Customizable per session."
        }
    },
    "output": "str — the consensus",
    "tags": ["ensemble", "multi-perspective"],
    "uses": ["generate"],
}

PERSPECTIVE_PROMPT = """You are a {perspective} thinker. Analyze the following task from your perspective.
Be specific and substantive.

Task:
{task}

Your {perspective} analysis:"""

SYNTHESIS_PROMPT = """The following perspectives have been gathered on a task.

{analyses_text}

Synthesize these perspectives into a single, well-reasoned response that incorporates the strongest points from each viewpoint.

Task:
{task}

Consensus:"""


async def discuss(
    llm,
    task: str,
    perspectives: list[str] | None = None
) -> str:
    perspectives = perspectives or ["analytical", "critical", "creative"]

    # Generate each perspective
    analyses = []
    for p in perspectives:
        analysis = await generate(llm, PERSPECTIVE_PROMPT.format(perspective=p, task=task))
        analyses.append((p, analysis))

    # Synthesize
    analyses_text = "\n\n".join(
        f"--- {p.capitalize()} perspective ---\n{a}" for p, a in analyses
    )
    consensus = await generate(llm, SYNTHESIS_PROMPT.format(
        analyses_text=analyses_text,
        task=task
    ))
    return consensus
