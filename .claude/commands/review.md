Run the human review workflow. This is the interactive counterpart to `/start`.

## Steps

1. List recent experiment runs: `uv run evaluate.py --report`
2. Pick a run to review (or review the latest): `uv run evaluate.py --review [RUN_ID]`
3. Rate each stored output interactively (y/n/s/q, with optional notes).
4. Once review is complete, analyze accumulated human signals across all reviewed runs:
   - Count total reviews, approval rate overall and per test case
   - List any **false positives** (judge avg >= 7 but human rejected) -- these suggest the judge rubric is too lenient or missing something
   - List any **hidden gems** (judge avg < 5 but human approved) -- these suggest the judge rubric is too strict or missing a quality the human values
   - Compare approval rate across recent runs to see if chain quality is trending up or down
5. If there are 3+ mismatches (false positives or hidden gems) in the last 10 reviews, suggest running `uv run evaluate.py --calibrate` for rubric adjustments.
6. Print a concise summary of findings.

Do NOT start the autonomous experiment loop. This command is for human-in-the-loop evaluation only.
