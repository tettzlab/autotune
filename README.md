# autotune -- Autonomous prompt chain optimizer

[![License: Apache-2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)

Autonomous prompt chain optimization. An AI agent edits a prompt chain, evaluates it against test cases, keeps improvements, discards regressions, and repeats -- indefinitely. You wake up to a log of experiments and a better chain.

> Inspired by [@karpathy's autoresearch](https://github.com/karpathy/autoresearch), which applies the same autonomous experiment loop to neural network training. autotune adapts that paradigm from model training to prompt engineering.

## How it works

The agent modifies a single file (`chain.py`) that defines a prompt chain. The task -- what the chain produces, from what inputs, and how it's judged -- is defined in `task.yaml`. Each experiment:

1. Edit `chain.py` (prompts, steps, models, dataflow -- anything)
2. Run evaluation against test cases
3. LLM-as-judge scores on binary criteria (see `criteria.md`)
4. Compute `adjusted_score = raw_score - lambda * chain_cost`
5. If improved: keep. Otherwise: revert.

All experiment artifacts (generated outputs, scores, criteria snapshots) are persisted to `experiments/` for human review and analysis.

## Project structure

```
task.yaml        -- task definition: inputs, output, judge role (swap this to change domains)
chain.py         -- prompt chain definition + execution (agent edits this)
llm.py           -- LLM call helper with token tracking (static)
evaluate.py      -- LLM-as-judge scoring harness (static)
store.py         -- experiment storage layer (static)
criteria.md      -- binary evaluation rubric (swap this per domain)
program.md       -- agent instructions
test_cases/      -- input data (swap this per domain)
experiments/     -- all experiment results, outputs, and reviews (git-tracked)
.claude/commands/start.md  -- /start slash command (autonomous loop)
.claude/commands/review.md    -- /review slash command (human-in-the-loop eval)
```

## Adapting to a new domain

To use autotune for a different task (e.g. technical blog posts, code reviews, customer support):

1. **Edit `task.yaml`** -- define your input fields, output description, and judge role
2. **Edit `criteria.md`** -- write binary (yes/no) evaluation criteria for your domain
3. **Replace `test_cases/*.json`** -- provide test cases with the input fields from `task.yaml`
4. **Edit `chain.py`** -- write an initial chain for your task (or let the agent start from scratch)

No framework code changes needed. The evaluation harness, storage, review, and calibration systems are fully generic.

## Quick start

**Requirements:** Python 3.11+, [uv](https://docs.astral.sh/uv/), an OpenAI and/or Anthropic API key.

```bash
# 1. Install dependencies
uv sync

# 2. Set up API keys
cp .env.example .env
# Edit .env with your keys

# 3. Run a single evaluation
uv run evaluate.py

# 4. Run with fewer cases for a quick test
uv run evaluate.py --cases 2

# 5. View all past experiment results
uv run evaluate.py --report

# 6. Review past outputs (human-in-the-loop)
uv run evaluate.py --review              # review latest run
uv run evaluate.py --review <RUN_ID>     # review a specific run
```

## Running the agent

Open Claude Code in this repo and run:

```
/start
```

Or prompt manually:

```
Read program.md and kick off a new experiment. Do the setup first.
```

The agent creates a branch, establishes a baseline, and enters the autonomous loop. Each cycle takes 1-5 minutes depending on chain complexity and model selection (reasoning models are slower). Expect 12-60 experiments/hour.

## Human review

After a research session, review the generated outputs:

```
/review
```

This lets you walk through stored outputs from any past experiment run and give thumbs-up/down ratings with optional notes. Reviews are stored per-case in `experiments/<run_id>/reviews/` and the agent reads them on the next `/start` run to guide experiment direction.

Mismatches between judge scores and your ratings (false positives, hidden gems) flag when the evaluation rubric is drifting. Run `uv run evaluate.py --calibrate` to propose rubric adjustments based on accumulated reviews.

## Experiment storage

Every evaluation run creates a self-contained directory:

```
experiments/<run_id>/
  meta.json         -- commit, branch, status, description
  task.yaml         -- snapshot of task definition used
  criteria.md       -- snapshot of criteria used for this run
  chain.py          -- snapshot of chain code used
  summary.json      -- aggregate scores, costs, timings
  cases/
    <label>/
      output.md     -- full generated output
      scores.json   -- per-criterion judge scores
  reviews/
    <label>.json    -- human review (added asynchronously)
```

This enables:
- Reviewing any past experiment's outputs at any time
- A/B comparison across experiments for the same test case
- Auditing that score improvements reflect real quality gains
- Tracking which criteria rubric was used for each run

## Available models

| Model | Provider | Reasoning efforts | Notes |
|---|---|---|---|
| `gpt-5-mini` | OpenAI | `minimal`, `low`, `medium`, `high` | Fast, cheap, good reasoning |
| `gpt-5-nano` | OpenAI | `minimal`, `low`, `medium`, `high` | Fastest, cheapest |
| `gpt-5.4` | OpenAI | `none`, `low`, `medium`, `high`, `xhigh` | Highest quality, expensive |
| `sonnet-4.6` | Anthropic | `low`, `medium`, `high` | High quality, adaptive thinking |
| `haiku-4.5` | Anthropic | _(not supported)_ | Fast, cheap |
| `opus-4.6` | Anthropic | `low`, `medium`, `high`, `max` | Highest quality, expensive |

## Evaluation metric

```
adjusted_score = raw_score - lambda * chain_cost
```

- `raw_score`: average pass rate on binary criteria across all test cases (normalized to 0-10)
- `chain_cost`: total API cost in USD for one evaluation run
- `lambda`: cost penalty weight (default 1.0, configurable via `COST_LAMBDA` env var)

This rewards both quality and cost efficiency.

## Design choices

- **Single file to modify.** The agent only touches `chain.py`. Diffs are reviewable, scope is contained.
- **Domain via config.** `task.yaml` defines what the chain does. Swap it (plus criteria + test cases) to apply autotune to any text generation task.
- **Guaranteed token tracking.** All LLM calls go through `llm.py`, which owns usage accounting. The agent cannot bypass it.
- **Cost-adjusted metric.** Raw quality isn't enough -- the metric penalizes expensive chains, pushing toward efficiency.
- **Durable experiment storage.** Every run's outputs, scores, and criteria are persisted as text files in `experiments/`. Nothing is lost between iterations.
- **Decoupled review.** Human review is independent of evaluation -- review any past run at any time without re-running the chain.
- **No GPU required.** Just API keys. Runs anywhere.

## License

Apache-2.0 — see [LICENSE](LICENSE).
