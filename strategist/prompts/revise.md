# Iteration {iteration}

Last iteration you proposed **5 workflow candidates**. The harness tested
each one at the anchors you declared and is reporting back below. Propose
**5 new candidates** this round.

<!-- reference:begin -->
## Bar to beat
{bar_to_beat}

The comparison with the reference (above and per candidate below) is the
state of the run: below the floor,
accuracy is what is missing; over the budget, cost; a tie or a small win
needs a wider margin on one dimension without losing the other, since
gains inside the noise are not picked.

<!-- reference:end -->

## About this benchmark
{benchmark_description}

## Results from your last 5 candidates

{candidate_results}

Each result block above contains, per (workflow, anchor):
- {blind_shown} The feedback split is training, the blind split
  validation (100 samples each, capped).
- Per-call breakdown: tier, mode, input tokens, output tokens, latency.
- Weighted cost total (sum of `tokens × tier_weight` across all calls).
- Up to 5 failed traces from the feedback set.

If a candidate does much better on feedback than on blind, it is
overfitting.

## Relevant blueprints from the library ({workflow_count} total; full measurement cards)
{search_results}

## Adjacent inspiration (different domains and reasoning patterns; one-liners)
{workflow_summary}

## Cards you requested last iteration
{requested_cards}

## Performance history
{performance_history}

## Your job

Propose a **fresh 5-candidate slate** (mix of `new` / `composed` / `reuse`).
`new` and `composed` candidates each require a `# WORKFLOW: <name>` block
in your response — the orchestrator writes them to
`strategist/workflows/<name>.py` before resolving the slate. Existing
library entries are never overwritten; pick fresh names for `new` and
`composed`, and point at existing names for `reuse`.

Read the failed traces carefully. Ask yourself:
- Which step introduced the error — classification, reasoning, verification,
  formatting?
- Is a tier/mode swap the fix, or is the workflow structure itself wrong?
- Is there a library workflow whose card would inform the next candidate?
  If so, request it via `read_cards`.

Anti-overfitting rules still apply:
- No hardcoded answers. No pattern-matching on specific question text.
- Name workflows by FUNCTION, not by benchmark.

## Read the card before composing

Every workflow in the library has a card in `strategist/cards/`. If you
put a workflow name in `builds_on`, you MUST have its card open. Put the
workflow name in `read_cards` this iteration so next iteration you see
the card. Requesting cards is free — request liberally. Never claim a
measured number ("card shows X achieves Y%") without the card in front
of you; attribution drift is the top failure mode.

## Mandatory slate composition (hard requirement — violating slates are rejected)

Your 5 candidates MUST include **at least one of each kind**:
- **`"new"`** (novel, no library-workflow backbone)
- **`"composed"`** (wraps a library workflow; import it by name)
- **`"reuse"`** (direct use of a library workflow; first-class, can win)

and **at least one of each target** (floor = the reference's accuracy, budget =
its cost): `"accuracy"` (beat the floor within the budget), `"efficiency"`
(hold the floor within 2 points at clearly lower cost), `"balanced"` (a little
of both). Kinds and targets are independent tags. The other 2 slots go
where your gap is (accuracy while below the floor, cost while over
budget).

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

If an existing library
workflow is already strong on this benchmark (cards, or the reference
block when one is shown), don't waste a slot reinventing it — use it
directly as a `reuse` candidate. Tag each candidate with `"kind"`,
`"target"` and `"builds_on"` (list of library workflow names for
composed/reuse; empty for new).

## Mandatory: experience notes this iteration

Every library workflow you used in the previous iteration's candidates, or
referenced via `read_cards` this round, MUST produce an `experience` entry
with concrete `observation` (what the feedback / card told you) and
concrete `recommendation` (when to use or avoid, stated as a rule). These
notes accumulate into the workflow cards — skipping them or writing
generic filler ("worked well", "fast") is rejected.

## Output

Zero or more `# WORKFLOW: <name>` fenced Python blocks (one per `new`
or `composed` candidate), followed by one `json` fenced block with the
slate. See the instructions above for the exact schema and naming rules.
