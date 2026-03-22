# autotune (Autonomous prompt chain optimizer)

This is an experiment to have the LLM optimize its own prompt chains.

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `mar21`). The branch `chainresearch/<tag>` must not already exist.
2. **Create the branch**: `git checkout -b chainresearch/<tag>` from current master.
3. **Read the in-scope files**: The repo is small. Read these files for full context:
   - `README.md` -- repository context.
   - `task.yaml` -- task definition: what the chain produces, input/output fields, judge role. Do not modify.
   - `llm.py` -- LLM call helper with token tracking. Do not modify.
   - `evaluate.py` -- evaluation harness with LLM-as-judge scoring. Do not modify.
   - `store.py` -- experiment storage layer. Do not modify.
   - `chain.py` -- **the file you modify.** Chain definition and execution logic.
   - `criteria.md` -- evaluation rubric. Do not modify (calibrated separately).
   - `test_cases/*.json` -- input data. Do not modify.
   - `experiments/` -- past experiment results and human reviews. Read for feedback signals.
4. **Verify API key**: Check that `OPENAI_API_KEY` (or `ANTHROPIC_API_KEY` if using Anthropic) is set in `.env`.
5. **Review past experiments**: Run `uv run evaluate.py --report` to see what's been tried. Check `experiments/*/reviews/` for human feedback.
6. **Confirm and go**: Confirm setup looks good.

Once you get confirmation, kick off the experimentation.

## What the system does

The task is defined in `task.yaml`. It specifies:
- **What** the chain produces (the output)
- **From what** inputs (named fields that test cases provide)
- **How** the judge evaluates (the judge's role/expertise)

`chain.py` defines a single async function `run_chain(inputs: dict[str, str]) -> str` that orchestrates the chain. You have full control over the implementation -- prompts, steps, models, dataflow, Python logic, whatever. The only constraint: **all LLM calls must use `call_llm()` from `llm.py`**.

The `inputs` dict contains the fields defined in `task.yaml`. Access them by name, e.g. `inputs["premise"]`. Check `task.yaml` for the current field names and descriptions.

### Available models
- `gpt-5-mini` -- OpenAI, fast and cheap
- `gpt-5-nano` -- OpenAI, fastest and cheapest
- `gpt-5.4` -- OpenAI, highest quality, expensive
- `sonnet-4.6` -- Anthropic, high quality
- `haiku-4.5` -- Anthropic, fast and cheap
- `opus-4.6` -- Anthropic, highest quality, expensive

### LLM call interface
```python
from llm import call_llm

response = await call_llm(
    messages=[{"role": "system", "content": "..."}, {"role": "user", "content": "..."}],
    model="gpt-5-mini",        # required
    reasoning_effort="medium",  # see effort table below
    temperature=0.7,            # Anthropic only; ignored for OpenAI reasoning models
    max_tokens=4096,            # max output tokens
)
# response is a string
```

Token usage and cost are tracked automatically by `llm.py`. You do NOT need to track them yourself.

### Reasoning effort
A key hyper-parameter. Higher effort may improve quality but costs more and is slower.

| Model | Valid efforts |
|---|---|
| `gpt-5-mini` | `minimal`, `low`, `medium` (default), `high` |
| `gpt-5-nano` | `minimal`, `low`, `medium` (default), `high` |
| `gpt-5.4` | `none`, `low`, `medium` (default), `high`, `xhigh` |
| `sonnet-4.6` | `low`, `medium` (default), `high` |
| `haiku-4.5` | _(not supported)_ |
| `opus-4.6` | `low`, `medium` (default), `high`, `max` |

## Evaluation

`evaluate.py` runs the chain against all test cases in `test_cases/` and scores each output using LLM-as-judge on binary (yes/no) criteria defined in `criteria.md`. Each "yes" = 1 point. Read `criteria.md` for the full list -- the evaluator loads it at runtime.

The overall score is the average pass rate across all criteria and all test cases (normalized to 0-10). Binary scoring is more reproducible and directly tells you which criteria failed, making it easier to target improvements.

**Calibration**: After collecting human reviews (`--review`), run `uv run evaluate.py --calibrate` to analyze judge-vs-human mismatches and propose revisions to `criteria.md`. This keeps the evaluation rubric aligned with human judgment over time.

## Experiment storage

Every evaluation run is persisted to `experiments/<run_id>/` with:
- `meta.json` -- run identity (commit, branch, status, description)
- `task.yaml` -- snapshot of the task definition used
- `criteria.md` -- snapshot of the criteria used for this run
- `chain.py` -- snapshot of the chain code used
- `summary.json` -- aggregate scores, costs, timings
- `cases/<label>/output.md` -- the full generated output
- `cases/<label>/scores.json` -- per-criterion judge scores and usage
- `reviews/<label>.json` -- human reviews (added asynchronously)

The `run_id` format is `<YYYYMMDD>T<HHMMSS>_<commit>`, printed in the evaluation output.

## Experimentation

**What you CAN do:**
- Modify `chain.py` -- this is the only file you edit. Everything is fair game:
  - Prompts, number of steps, step order
  - Model selection per step
  - Reasoning effort, temperature, max_tokens per step
  - Dataflow: sequential, branching, parallel (asyncio.gather), critique loops
  - Any Python logic for orchestrating the chain
  - Import any stdlib module

**What you CANNOT do:**
- Modify `llm.py`, `evaluate.py`, `store.py`, `task.yaml`, or `test_cases/*.json`.
- Install new packages or add dependencies.
- Call LLM APIs directly -- you MUST use `call_llm()` from `llm.py`.
- Modify the evaluation rubric or judge prompt.

**The goal is simple: get the highest `adjusted_score`.** This is `score - lambda * chain_cost` where lambda defaults to 1.0. It rewards quality but penalizes expensive chains. Both quality and cost efficiency matter.

**Simplicity criterion**: All else being equal, simpler is better.

**The first run**: Always establish the baseline first by running evaluation as is.

## Output format

The evaluation script prints a summary like this:

```
---
run_id:           20260321T143022_a1b2c3f
score:            8.8000
adjusted_score:   8.7541
cost_lambda:      1.0
  three_act_structure   100%  pass
  character_arcs        100%  pass
  central_conflict       80%  pass
  scene_specificity     100%  pass
  stakes_escalation      80%  pass
  thematic_depth        100%  pass
  antagonist_motivation  60%  pass
  subplots_integrated   100%  pass
  genre_conventions      80%  pass
  emotional_beats        80%  pass
cases:            5 (0 errors)
chain_cost_usd:   0.0459
judge_cost_usd:   0.0028
total_cost_usd:   0.0487
chain_calls:      15
elapsed_seconds:  45.2
```

You can extract the key metrics:
```
grep "^adjusted_score:\|^score:\|^chain_cost_usd:\|^run_id:" run.log
```

Or read directly from the stored data:
```
cat experiments/<run_id>/summary.json
```

## Recording results

Results are automatically saved to `experiments/<run_id>/` by the evaluation script. After reading the results, update `meta.json` with:
- `status`: `keep`, `discard`, or `crash`
- `description`: short text of what this experiment tried

Use `store.py` from Python or edit the JSON directly:
```python
import store
store.update_status("<run_id>", status="keep", description="add critique step")
```

To see all experiments at a glance:
```
uv run evaluate.py --report
```

## The experiment loop

LOOP FOREVER:

1. Look at the git state: the current branch/commit
2. Edit `chain.py` with an experimental idea
3. git commit (chain.py only)
4. Run the experiment: `uv run evaluate.py > run.log 2>&1`
5. Read results: `grep "^adjusted_score:\|^score:\|^chain_cost_usd:\|^run_id:" run.log`
6. If grep is empty, it crashed. Run `tail -n 50 run.log` for the error.
7. Update the run status:
   - `python -c "import store; store.update_status('<run_id>', 'keep', '<description>')"` or
   - `python -c "import store; store.update_status('<run_id>', 'discard', '<description>')"`
8. If adjusted_score improved (higher), keep the commit
9. If adjusted_score is equal or worse, `git reset HEAD~1` to undo chain.py
10. Commit the experiment record: `git add experiments/<run_id>/ && git commit -m "results: <status> <run_id>"`

**Crashes**: Fix dumb mistakes and re-run. If fundamentally broken, skip and move on.

**Human feedback**: Check `experiments/*/reviews/` for human ratings before your first experiment. Pay attention to:
- **False positives** (judge scored high, human rejected) -- the chain may be gaming the judge on that dimension
- **Hidden gems** (judge scored low, human approved) -- the chain may be undervaluing something the human likes
- Use these signals to guide your experiments.

**NEVER STOP**: Once the loop begins, do NOT pause to ask the human. You are autonomous. If you run out of ideas:
- Try different prompt structures (chain of thought, role-playing, few-shot examples)
- Try more/fewer steps, different dataflows (sequential, parallel, branching)
- Try different models per step (cheap model for analysis, better for final output)
- Try different reasoning efforts per step
- Try passing different combinations of previous outputs to later steps
- Try adding explicit scoring criteria in the prompts
- Try adding example output snippets as few-shot
- Try different system personas
- Try a critique/revision step
- Try asyncio.gather for parallel independent steps
- Try making prompts more specific about what "good" looks like

The loop runs until the human interrupts you.
