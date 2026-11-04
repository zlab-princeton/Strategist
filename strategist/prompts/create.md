# Iteration 1

Propose **5 workflow candidates** for the **{benchmark_name}** benchmark. The
harness will test each one independently on the training + validation splits
(100 samples each, capped; small benchmarks scale down proportionally) at the
configurations you declare.

<!-- reference:begin -->
## Bar to beat
{bar_to_beat}

Read these numbers before proposing anything. The reference's accuracy is
the floor you must not fall below; its cost is the budget you must not
exceed; within 2 points / 2% is a tie, and a win beats it beyond that on one
of the two and holds the other. The harness measures the reference itself
(its exact blueprint at its anchor) in this iteration, so it does not
need a slot.
<!-- reference:end -->

## About this benchmark
{benchmark_description}

## Example questions (from the feedback set — answers visible)
{sample_questions}

## Relevant blueprints from the library ({workflow_count} total; full measurement cards)
{search_results}

## Adjacent inspiration (different domains and reasoning patterns; one-liners)
{workflow_summary}

## Cards you requested
{requested_cards}

## Performance history
{performance_history}

## Your job

Propose a **5-candidate slate** worth measuring on this benchmark. The
slate is a mix of `new` (write new workflows to the library), `composed`
(write new workflows that wrap existing library entries), and `reuse`
(point at existing library entries unchanged). Library promotion is
automatic — every `new` and `composed` candidate requires a
`# WORKFLOW: <name>` fenced block in your response; each is written to
`strategist/workflows/<name>.py` before the slate is resolved.

See the instructions above for `llms`, blocks, tier/mode trade-offs, and the
exact output schema.

For each candidate, declare up to 3 `tier_mode_anchors` in the JSON. Each
anchor is either:
- `"multi"` — run the workflow with the full `llms` dict so it can mix tiers.
- `"{tier}/{mode}"` — redirect every `llms` entry to one single (tier, mode)
  pair. Useful when you want to learn whether the workflow's structure helps
  at a specific capability level.

Fewer anchors = fewer samples burned. Pick wisely. If you have no strong
intuition for tier/mode, one `multi` anchor is enough.

## Anti-overfitting rules

- NEVER hardcode answers, lookup tables, or pattern-match on specific
  question text. Blind eval will catch it.
- Name workflows by FUNCTION, not by benchmark (plus the task-family suffix
  when the prompts are task-specific, see the instructions above).
- One clear idea per candidate, not a catch-all stack — but give the idea
  the stages it needs (decomposition, tool use, voting, verification,
  routing, blocks). A thin wrapper that adds one call to a library blueprint is the
  floor, not the goal.

## Read the card before composing

Every workflow in the library has a card in `strategist/cards/`. If you
put a workflow name in `builds_on`, you MUST have its card open. Put the
workflow name in `read_cards` this iteration so next iteration you see
the card. Requesting cards is free — request liberally. Don't build on
a workflow whose card you haven't read; names are ambiguous and your
training-data memory of similar workflows is not reliable enough.

## Mandatory slate composition (hard requirement — violating slates are rejected)

Your 5 candidates MUST include **at least one of each kind**:
- **`"new"`** (novel, no library-workflow backbone)
- **`"composed"`** (wraps a library workflow with a novel improvement —
  import the wrapped workflow by name)
- **`"reuse"`** (directly use a library workflow as the candidate; a
  first-class candidate that can win)

and **at least one of each target** (floor = the reference's accuracy, budget = its cost):
- **`"accuracy"`** (beat the floor within the budget)
- **`"efficiency"`** (hold the floor within 2 points at clearly lower cost)
- **`"balanced"`** (a little of both: above the floor and under the budget)

**Innovation.** At least one `new` candidate must be truly innovative: be
creative and explore something no one has done, while staying analytical —
learn from the failures you have seen and let that analysis drive the
invention. Not another verify step with new wording. Innovative does not
mean task-specific: invent a general reasoning strategy, one that any task
needing this kind of reasoning can use, and keep the task-specific knowledge
in its prompts (`prompts` slots), not in its structure — the library is
shared across many tasks. At least one `composed` candidate must be innovative in
the same way, building on an existing blueprint. The remaining slots are
yours to arrange.

Structure of this run's blueprints so far: {run_structure}

Kinds and targets are independent tags; the two free slots go where your
gap is (accuracy while below the floor, cost while over budget). A slate
missing a kind or a target, omitting a required field, or structurally
duplicating an existing blueprint is rejected and you will be re-prompted.

Tag each candidate with `"kind"`: `"new"` / `"composed"` / `"reuse"` and
`"target"`: `"accuracy"` / `"efficiency"` / `"balanced"`. Set
`"builds_on": [...]` to the library workflow names used (required for
composed/reuse).

## Mandatory: experience notes

Every library workflow you referenced (inside a candidate, as inspiration,
or via `read_cards`) MUST get an `experience` entry in the JSON. Each
entry needs a concrete `observation` (what the numbers or card told you)
and a concrete `recommendation` (when to use or avoid it). Generic one-
liners ("good for math", "fast") are rejected. Cite specific accuracies,
cost ratios, or failure modes from the cards (or the reference block when one is shown).

## Optional: self-generated tests

You may include `add_tests` in the JSON to build your own
overfitting detector. Not required.

## Output

Zero or more `# WORKFLOW: <name>` fenced Python blocks (one per `new`
or `composed` candidate), followed by one `json` fenced block with the
slate. See the instructions above for the exact schema and naming rules.
