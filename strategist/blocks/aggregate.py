from strategist.basic import generate

META = {
    "name": "aggregate",
    "description": "Merge multiple results into a single coherent output.",
    "when_to_use": "After solving sub-tasks independently. Combine the pieces into one answer.",
    "inputs": {
        "results": {
            "type": "list[str]",
            "required": True,
            "description": "The individual results to merge"
        },
        "task": {
            "type": "str",
            "required": False,
            "default": "",
            "description": "The original task, for context when merging"
        }
    },
    "output": "str — the merged result",
    "tags": ["aggregation", "synthesis"],
    "uses": ["generate"],
}

PROMPT = """Merge the following {n} results into a single coherent response.
Resolve any contradictions. Eliminate redundancy. Produce a unified answer.
{task_line}
{results_text}

Merged result:"""


async def aggregate(llm, results: list[str], task: str = "") -> str:
    task_line = f"\nOriginal task: {task}" if task else ""
    results_text = "\n\n".join(
        f"--- Result {i+1} ---\n{r}" for i, r in enumerate(results)
    )
    response = await generate(llm, PROMPT.format(
        n=len(results),
        task_line=task_line,
        results_text=results_text
    ))
    return response
