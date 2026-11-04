# Iteration {iteration}

The harness measured the 5 candidates of your last answer; the results are
below. Everything earlier in this conversation still holds: the
rules, the output format, the benchmark, the baseline table and your own
earlier reasoning. Propose **5 new candidates** this round.

<!-- reference:begin -->
## Bar to beat (updated)
{bar_to_beat}

The comparison with the reference (above and per candidate below) is the
state of the run: below the floor,
accuracy is what is missing; over the budget, cost; a tie or a small win
needs a wider margin on one dimension without losing the other, since
gains inside the noise are not picked.
<!-- reference:end -->

## Results from your last 5 candidates

{candidate_results}

## Relevant blueprints from the library ({workflow_count} total; full measurement cards)
{search_results}

## Adjacent inspiration (one-liners)
{workflow_summary}

## Cards you requested last iteration
{requested_cards}

## Performance history (every candidate of this run, one line each)
{performance_history}

## Your answer

Use what you have learned over this run, and say in each `why` what
earlier result the candidate builds on.

Every candidate measured so far is already in the final comparison with
the reference. Proposing one again with the same blueprint, anchor and
prompts measures nothing: the harness reuses its earlier numbers and the
slot is wasted. Each candidate should be something not yet measured in
this run (a new blueprint, or a library one at another anchor or with
other prompts).

Same rules as before (violating slates are rejected):
- 5 candidates, at least one each of kind `new`, `composed`, `reuse` and of
  target `accuracy`, `efficiency`, `balanced`; at least one `new` and one
  `composed` candidate truly innovative (a general reasoning strategy, with
  the task-specific knowledge in its `prompts` slots).
- A `# WORKFLOW: <name>` block for every `new` / `composed` candidate, with
  a fresh name; `reuse` points at an existing name.
- An `experience` entry, with a concrete `observation` and
  `recommendation`, for every blueprint measured last round and every
  blueprint in this round's `read_cards`.
- Put in `read_cards` every blueprint you will build on next round.

Structure of this run's blueprints so far: {run_structure}

Output: zero or more `# WORKFLOW: <name>` fenced Python blocks, then one
`json` fenced block with the slate, using exactly these keys (`prompts` is
optional):

```json
{
  "candidates": [
    {"name": "...", "kind": "new|composed|reuse", "target": "accuracy|efficiency|balanced",
     "tier_mode_anchors": ["multi"], "why": "...", "builds_on": [], "prompts": {"p1": "..."}}
  ],
  "search_queries": ["..."],
  "read_cards": ["..."],
  "experience": [{"workflow": "...", "observation": "...", "recommendation": "..."}]
}
```
