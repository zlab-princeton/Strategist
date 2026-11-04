# Task files

A search reads its development questions from `--val-set` and, when
`strategist/data/<task>_test.jsonl` exists, runs the final pick on that file too.
Nothing is bundled; put your own files here (they are gitignored). One JSON object
per line:

```json
{"task": "If 3x + 5 = 20, what is x?", "answer": "5"}
```

- `task`: the full prompt the strategy receives.
- `answer`: the gold answer as a string. Add `answers` (a list) when several
  references are accepted. Code tasks add `entry_point` and `test_code`; their
  outputs are run against the tests instead of compared as text.

`strategist/scoring.py` grades every other output by the gold answer's form: yes/no, a
choice letter, a choice index, a top-5 ranking, a number, or text, each matched
exactly after answer extraction and normalisation.
