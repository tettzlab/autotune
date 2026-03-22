# Architecture

## In a Nutshell

autotune is a self-improving prompt chain optimizer. An AI agent edits a prompt chain (`chain.py`), runs it against test cases, scores the output with an LLM judge, and keeps changes that improve the score. Everything else -- the LLM wrapper, the evaluation harness, the storage layer, the test data -- is frozen. The agent can only turn one knob: how the chain is wired.

Think of it as gradient descent for prompt engineering: the "loss function" is `adjusted_score = raw_score - lambda * cost`, the "parameters" are the prompts/steps/models/dataflow in `chain.py`, and the "optimizer" is an LLM agent making educated guesses about what to try next.

The concrete task is defined in `task.yaml` -- the included sample generates novel plot outlines from a premise, genre, and character descriptions, but the framework is domain-agnostic. Swap `task.yaml`, `criteria.md`, and `test_cases/` to apply it to any text generation task.

## System Overview

```
                         +-----------+
                         | program.md|  <-- agent instructions
                         +-----+-----+
                               |
                         +-----v-----+
                         |   Agent   |  <-- Claude Code (/start)
                         +-----+-----+
                               |
                          edits only
                               |
                         +-----v-----+
                 +------>| chain.py  |  <-- the ONE mutable file
                 |       +-----+-----+
                 |             |
                 |       run_chain(inputs)
                 |             |
                 |       +-----v-----+
                 |       |  llm.py   |  <-- call_llm() + token tracking
                 |       +-----------+
                 |             |
                 |       +-----v-----+
            revert if    |evaluate.py|  <-- runs chain on test_cases/
            worse        +-----+-----+     scores with LLM-as-judge
                 |             |
                 |       +-----v-----+
                 |       | store.py  |  <-- persists to experiments/
                 |       +-----+-----+
                 |             |
                 +-------------+
                     loop forever
```

## Module-by-Module Breakdown

### `task.yaml` -- Task Definition (Swap per Domain)

Defines what the chain produces and how it's evaluated, without any code changes:

```yaml
name: "Novel Plot Outline"
description: "Generate a detailed, compelling plot outline for a novel..."

inputs:
  - name: premise
    description: "A short description of the story's core concept, setting, and central conflict"
  - name: genre
    description: "The genre and tone (e.g. literary fiction, sci-fi thriller, dark fantasy)"
  - name: characters
    description: "Markdown describing the main characters, their backgrounds, motivations, and relationships"

output:
  name: plot_outline
  description: "A detailed novel plot outline in markdown"

judge_role: "You are an expert fiction editor and story structure analyst..."
```

The evaluation harness reads this at startup and:
- Builds the judge system prompt from `judge_role` + `inputs` descriptions + criteria
- Extracts the right fields from test case JSON files using `inputs[].name`
- Passes `inputs` dict to `chain.run_chain()`

To adapt autotune to a new domain, change `task.yaml` + `criteria.md` + `test_cases/`. No framework code changes.

### `chain.py` -- The Prompt Chain (Mutable)

The only file the agent modifies. Defines a single async function:

```python
async def run_chain(inputs: dict[str, str]) -> str
```

Takes a dict of named input fields (as defined in `task.yaml`), orchestrates a multi-step LLM pipeline, and returns the final output text. For the included novel plot outline task, `inputs` contains `"premise"`, `"genre"`, and `"characters"`.

**What's fair game for modification:**
- Prompts (system messages, user messages, structure)
- Number and order of steps
- Model selection per step (`gpt-5-mini`, `gpt-5-nano`, `gpt-5.4`, `sonnet-4.6`, `haiku-4.5`, `opus-4.6`)
- Reasoning effort per step (`low`, `medium`, `high`, etc.)
- Temperature, max_tokens
- Dataflow: sequential, branching, parallel (`asyncio.gather`), critique loops
- Any Python logic for orchestration

**The one constraint:** all LLM calls must use `call_llm()` from `llm.py`.

**Current architecture** (as of latest commit):
1. **Analyze** -- extract story structure, character arcs, thematic questions, act breakdown (gpt-5-nano, low effort)
2. **Generate** -- write the full plot outline using analysis + raw inputs (gpt-5-mini, high effort)
3. **Critique & Revise** -- self-critique against the 10 criteria, then output a revised outline (gpt-5-mini, medium effort)

Step 1 is cheap reconnaissance. Step 2 is the expensive creative work. Step 3 is a self-correction loop. The chain is fully sequential -- each step depends on the previous.

### `llm.py` -- LLM Abstraction Layer (Static)

A thin wrapper around [PydanticAI](https://ai.pydantic.dev/) that provides a single function:

```python
async def call_llm(
    messages: list[dict[str, str]],
    model: str = "gpt-5-mini",
    temperature: float = 0.7,
    max_tokens: int = 4096,
    reasoning_effort: str = "medium",
    output_type: type[T] | None = None,  # structured output via Pydantic model
) -> str | T
```

**Key internals:**

- **Model registry** (`MODELS` dict): Maps friendly names to PydanticAI model IDs, provider info, pricing (input/output per 1M tokens), and valid reasoning efforts. Supports OpenAI (via Responses API) and Anthropic.

- **Usage tracking** (`LLMUsage` dataclass + `contextvars`): Every call logs input/output tokens, cost, duration, and model name. Uses `contextvars.ContextVar` so parallel async tasks get isolated tracking. Accessed via `get_usage()` / `reset_usage()`.

- **Cost estimation**: `(input_tokens / 1M) * input_price + (output_tokens / 1M) * output_price`.

- **Provider-specific settings**: OpenAI gets `openai_reasoning_effort`; Anthropic gets `temperature` + `anthropic_effort`. OpenAI reasoning models don't support temperature.

- **Env loading**: Reads `.env` file at import time for API keys.

- **Structured output**: Pass a Pydantic `BaseModel` subclass as `output_type` to get typed responses instead of raw strings. PydanticAI handles the schema enforcement.

### `evaluate.py` -- Evaluation Harness (Static)

Runs the chain against test cases and scores outputs. The central CLI for the system.

**Core flow:**

```
load_test_cases() --> for each case:
    reset_usage()
    chain.run_chain(inputs) --> output
    snapshot chain usage
    reset_usage()
    judge_output(inputs, output) --> {criterion: 0|1}
    snapshot judge usage
    store results
--> aggregate into summary
```

**The LLM-as-judge:**

- Loads `criteria.md` + `task.yaml` at import time to build the judge system prompt dynamically
- The judge role, input descriptions, and output name come from `task.yaml`
- Sends all task inputs + generated output to the judge model (default: `gpt-5-mini`)
- Judge returns JSON: `{"scores": [1, 0, 1, ...], "total": N}`
- Each criterion is binary (yes=1, no=0), max score = 10

**Concurrency:** Uses `asyncio.Semaphore` (default 3) with `contextvars.copy_context()` per task so each case gets isolated LLM usage tracking.

**The adjusted score:**

```
adjusted_score = raw_score - COST_LAMBDA * chain_cost_usd
```

`raw_score` is the average pass rate across all criteria and cases, normalized to 0-10. `COST_LAMBDA` defaults to 1.0. This means $1 of chain cost wipes out 1 point of score -- a strong incentive for efficiency.

**CLI modes:**
| Flag | Mode |
|---|---|
| (none) | Run evaluation, save results |
| `--cases N` | Limit to first N test cases |
| `--verbose` | Print output excerpts |
| `--review [RUN_ID]` | Interactive human review of stored outputs |
| `--calibrate` | Analyze human reviews vs judge scores, propose criteria revisions |
| `--report` | Summary table across all experiment runs |

**Calibration flow** (`--calibrate`):
1. Load all human reviews across all runs
2. Compute per-criterion pass rates for human-approved vs human-rejected outputs
3. Flag "loose" criteria (pass rate > 50% on rejected outputs) and "strict" criteria (fail rate > 50% on approved outputs)
4. Send analysis + current `criteria.md` to an LLM to propose revised criteria
5. User confirms or rejects the revision

### `store.py` -- Experiment Storage (Static)

Handles all filesystem I/O for experiment data. Provides a clean API so no other module touches `experiments/` directly.

**Directory layout per run:**

```
experiments/<run_id>/
    meta.json           -- identity: run_id, timestamp, commit, branch, status, description
    task.yaml           -- snapshot of task definition used
    criteria.md         -- snapshot of criteria used for this run
    chain.py            -- snapshot of chain code used
    summary.json        -- aggregate: score, adjusted_score, costs, pass rates
    cases/
        <label>/
            output.md -- the generated output
            scores.json -- per-criterion scores + chain/judge usage
    reviews/
        <label>.json    -- human review (approved bool + notes)
```

**Run ID format:** `<YYYYMMDD>T<HHMMSS>_<short-commit-hash>`

**Key design decisions:**
- **Atomic writes**: Uses `tempfile.mkstemp` + `os.replace` for crash safety
- **Git info capture**: Records commit hash and branch at run creation time
- **Decoupled reviews**: Reviews are added asynchronously, independent of evaluation runs
- **Chronological listing**: `list_runs()` returns sorted by timestamp

**API surface:**

| Function | Purpose |
|---|---|
| `create_run()` | Create run directory, write `meta.json` |
| `save_criteria_snapshot()` | Copy current criteria into run |
| `save_case_result()` | Write output + scores for one case |
| `save_summary()` | Write aggregate summary |
| `update_status()` | Set `keep`/`discard`/`crash` + description |
| `list_runs()` / `latest_run()` | Enumerate runs |
| `load_meta()` / `load_summary()` / `load_case_result()` | Read stored data |
| `save_review()` / `load_reviews()` / `load_all_reviews()` | Human review CRUD |
| `list_unreviewed_runs()` | Find runs needing human review |

### `criteria.md` -- Evaluation Rubric

10 binary criteria for judging output quality (criteria for the novel plot outline task):

| # | Criterion | What it checks |
|---|---|---|
| 1 | `three_act_structure` | Clear three-act structure with identifiable act breaks |
| 2 | `character_arcs` | Each major character has distinct arc (want, goal, transformation) |
| 3 | `central_conflict` | Central conflict drives plot from inciting incident to climax |
| 4 | `scene_specificity` | At least 10 specific key scenes with concrete actions |
| 5 | `stakes_escalation` | Stakes escalate with 3+ worsening moments |
| 6 | `thematic_depth` | Theme woven through events and choices, not just stated |
| 7 | `antagonist_motivation` | Antagonist has coherent, non-trivial motivation |
| 8 | `subplots_integrated` | At least 2 subplots connected to main plot |
| 9 | `genre_conventions` | Genre conventions respected with a fresh element |
| 10 | `emotional_beats` | Specific emotional turning points, not just plot mechanics |

These criteria are intentionally binary to maximize reproducibility. The rubric itself can evolve via the `--calibrate` flow.

### `test_cases/` -- Input Data (Static)

5 JSON files, each containing a `label`, `premise`, `genre`, and `characters`:

| # | Label | Genre |
|---|---|---|
| 01 | Sci-fi colony ship mutiny | Hard sci-fi / political thriller |
| 02 | Literary fiction - grief and forgiveness | Literary fiction |
| 03 | Dark fantasy revolution | Dark fantasy |
| 04 | Techno-thriller whistleblower | Techno-thriller |
| 05 | Magical realism - migration and identity | Magical realism |

Each test case includes a detailed premise, genre/tone description, and rich character profiles with backgrounds, motivations, flaws, and relationships. This gives the chain enough material to produce a genuinely specific output.

### `program.md` -- Agent Instructions

The instruction manual for the agent. Defines:
- **Setup protocol**: branch naming, file reading, API key verification
- **What the system does**: references `task.yaml` for task definition
- **Available models**: with pricing and reasoning effort tables
- **Evaluation mechanics**: adjusted score formula, criteria overview
- **Experiment storage**: directory layout, status tracking
- **The experiment loop**: edit -> commit -> evaluate -> keep/revert -> record -> repeat
- **Constraints**: only `chain.py` is mutable, must use `call_llm()`, no new dependencies
- **Idea bank**: prompt structures, dataflow patterns, model mixing, etc.

Key rule: **"NEVER STOP"** -- once the loop begins, the agent runs autonomously until interrupted.

### `.claude/commands/` -- Slash Commands

| Command | Purpose |
|---|---|
| `/start` | Start the autonomous experiment loop (reads `program.md`) |
| `/review` | Interactive human review of stored outputs |
| `/fix` | Address specific issues with a structured validate-plan-apply-verify workflow |

## Data Flow

### The Experiment Loop

```
1. Agent edits chain.py
2. git commit (chain.py only)
3. Run: uv run evaluate.py > run.log 2>&1
4. Read results: grep adjusted_score/score/cost from run.log
5. Update run status (keep/discard) via store.update_status()
6. If improved: keep commit. If not: git reset HEAD~1
7. Commit experiment record: git add experiments/<run_id>/ && git commit
8. Goto 1
```

### Single Evaluation Run

```
evaluate.py::run_evaluation()
    |
    +-- store.create_run()                  # create experiments/<run_id>/
    +-- store.save_criteria_snapshot()       # freeze criteria.md
    |
    +-- for each test case (concurrent):
    |       |
    |       +-- chain.run_chain(inputs)     # inputs from task.yaml field names
    |       |       |
    |       |       +-- call_llm() x N     # tracked by llm.py
    |       |       |
    |       |       +-- returns output string
    |       |
    |       +-- judge_output(inputs, output)
    |       |       |
    |       |       +-- call_llm()          # single judge call
    |       |       |
    |       |       +-- returns {criterion: 0|1}
    |       |
    |       +-- store.save_case_result()
    |
    +-- aggregate scores, costs, timings
    +-- store.save_summary()
    +-- print results
```

### Token Usage Isolation

```
asyncio.Semaphore(3)
    |
    +-- Task 1: copy_context() -> isolated ContextVar[LLMUsage]
    |       reset_usage() -> chain calls -> snapshot -> reset -> judge call -> snapshot
    |
    +-- Task 2: copy_context() -> isolated ContextVar[LLMUsage]
    |       reset_usage() -> chain calls -> snapshot -> reset -> judge call -> snapshot
    |
    +-- Task 3: copy_context() -> isolated ContextVar[LLMUsage]
            reset_usage() -> chain calls -> snapshot -> reset -> judge call -> snapshot
```

Each concurrent test case gets its own `LLMUsage` accumulator via `contextvars.copy_context()`, so token counts and costs don't bleed between cases.

## Practical Usage

### First-Time Setup

```bash
# Clone and install
uv sync

# Set API keys
cp .env.example .env
# Edit .env: OPENAI_API_KEY=sk-... and/or ANTHROPIC_API_KEY=sk-ant-...

# Run baseline evaluation
uv run evaluate.py
```

### Running the Autonomous Agent

```bash
# In Claude Code:
/start
```

This reads `program.md`, creates a branch (`chainresearch/<tag>`), establishes a baseline, and enters the infinite experiment loop. The agent modifies `chain.py`, evaluates, keeps improvements, reverts regressions, and commits experiment records.

### Reviewing Results

```bash
# See all experiments
uv run evaluate.py --report

# Review latest run interactively
uv run evaluate.py --review

# Review a specific run
uv run evaluate.py --review 20260321T143022_a1b2c3f

# Calibrate criteria based on human reviews
uv run evaluate.py --calibrate
```

### Writing Your Own Chain

Edit `chain.py`. The only rules:

1. Export `async def run_chain(inputs: dict[str, str]) -> str`
2. Use `from llm import call_llm` for all LLM calls
3. Return the output as a string

Example minimal chain:

```python
from llm import call_llm

async def run_chain(inputs: dict[str, str]) -> str:
    # Access input fields by name (defined in task.yaml)
    context = "\n\n".join(f"## {k}\n{v}" for k, v in inputs.items())
    return await call_llm(
        messages=[
            {"role": "system", "content": "Generate the requested output."},
            {"role": "user", "content": context},
        ],
        model="gpt-5-mini",
        max_tokens=4096,
    )
```

Example multi-step chain with parallel steps:

```python
import asyncio
from llm import call_llm

async def run_chain(inputs: dict[str, str]) -> str:
    premise, genre, characters = inputs["premise"], inputs["genre"], inputs["characters"]

    # Parallel analysis
    structure_task = call_llm(messages=[...], model="gpt-5-nano", reasoning_effort="low")
    character_task = call_llm(messages=[...], model="gpt-5-nano", reasoning_effort="low")
    structure_analysis, character_analysis = await asyncio.gather(structure_task, character_task)

    # Sequential synthesis
    return await call_llm(
        messages=[...use both analyses...],
        model="gpt-5-mini",
        reasoning_effort="high",
        max_tokens=8192,
    )
```

## Key Design Principles

1. **Single point of mutation.** Only `chain.py` changes during experiments. Everything else is infrastructure. This makes experiments reviewable (just diff `chain.py`) and safe (you can't break the evaluation by accident).

2. **Domain via config.** `task.yaml` defines inputs, output, and judge role. The evaluation harness, storage, review, and calibration are fully generic. Swap `task.yaml` + `criteria.md` + `test_cases/` to apply autotune to any text generation task -- no code changes needed.

3. **Guaranteed accounting.** All LLM calls go through `call_llm()`, which owns token and cost tracking. The agent cannot game the cost metric.

4. **Cost-aware optimization.** The adjusted score penalizes expensive chains. A chain that scores 10/10 but costs $5 per run gets adjusted to 5.0 -- worse than a chain scoring 8/10 at $0.05 (adjusted 7.95). This pushes toward efficient architectures.

5. **Durable experiment records.** Every run is a self-contained directory with outputs, scores, criteria snapshot, and metadata. Nothing is lost between iterations. You can review any past experiment at any time.

6. **Decoupled human feedback.** Reviews are added asynchronously, separate from evaluation. The `--calibrate` flow closes the loop by adjusting the rubric based on judge-vs-human mismatches.

7. **No GPU, no infra.** Just API keys and Python. Runs on any machine with network access.
