"""
Evaluation harness for prompt chain research.
Runs the chain against all test cases and scores with LLM-as-judge.
All experiment artifacts are persisted to experiments/ via store.py.

Usage:
    uv run evaluate.py                    # run evaluation, save to experiments/
    uv run evaluate.py --cases 3          # run only first 3 cases
    uv run evaluate.py --verbose          # print per-case output excerpts
    uv run evaluate.py --review [RUN_ID]  # review stored outputs (default: latest)
    uv run evaluate.py --calibrate        # analyze reviews and propose criteria revisions
    uv run evaluate.py --report           # print summary table across all runs
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import contextvars
import importlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import store
from llm import LLMUsage, call_llm, get_usage, reset_usage

# ---------------------------------------------------------------------------
# Task configuration (loaded from task.yaml)
# ---------------------------------------------------------------------------

TASK_FILE = Path(__file__).parent / "task.yaml"


def _load_task() -> dict[str, Any]:
    """Load task.yaml. Uses a simple YAML subset parser to avoid extra deps."""
    if not TASK_FILE.exists():
        print(f"ERROR: {TASK_FILE} not found. Create it to define your task.")
        sys.exit(1)
    try:
        import yaml  # type: ignore[import-untyped]

        return yaml.safe_load(TASK_FILE.read_text())  # type: ignore[no-any-return]
    except ImportError:
        pass
    # Minimal fallback parser for our simple task.yaml structure
    return _parse_simple_yaml(TASK_FILE.read_text())


def _parse_simple_yaml(text: str) -> dict[str, Any]:
    """Parse the subset of YAML used in task.yaml (no external deps)."""
    result: dict[str, Any] = {}
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            i += 1
            continue
        # Top-level key
        if not line.startswith(" ") and ":" in stripped:
            key, _, val = stripped.partition(":")
            key = key.strip()
            val = val.strip()
            if val.startswith(">"):
                # Multi-line folded scalar
                parts: list[str] = []
                i += 1
                while i < len(lines) and (lines[i].startswith("  ") or not lines[i].strip()):
                    if lines[i].strip():
                        parts.append(lines[i].strip())
                    i += 1
                result[key] = " ".join(parts)
                continue
            if val == "":
                # Could be a list or nested mapping
                items: list[Any] = []
                nested: dict[str, Any] = {}
                i += 1
                while i < len(lines) and (lines[i].startswith("  ") or not lines[i].strip()):
                    sub = lines[i].strip()
                    if not sub or sub.startswith("#"):
                        i += 1
                        continue
                    if sub.startswith("- "):
                        item_line = sub[2:].strip()
                        if ":" in item_line:
                            # List of dicts
                            item_dict: dict[str, str] = {}
                            k2, _, v2 = item_line.partition(":")
                            item_dict[k2.strip()] = v2.strip().strip('"').strip("'")
                            i += 1
                            while i < len(lines) and lines[i].startswith("    ") and not lines[i].strip().startswith("- "):
                                sk, _, sv = lines[i].strip().partition(":")
                                item_dict[sk.strip()] = sv.strip().strip('"').strip("'")
                                i += 1
                            items.append(item_dict)
                            continue
                        else:
                            items.append(item_line)
                    elif ":" in sub:
                        k3, _, v3 = sub.partition(":")
                        v3 = v3.strip().strip('"').strip("'")
                        if v3.startswith(">"):
                            parts2: list[str] = []
                            i += 1
                            while i < len(lines) and (lines[i].startswith("    ") or not lines[i].strip()):
                                if lines[i].strip():
                                    parts2.append(lines[i].strip())
                                i += 1
                            nested[k3.strip()] = " ".join(parts2)
                            continue
                        nested[k3.strip()] = v3
                    i += 1
                result[key] = items if items else nested
                continue
            # Simple scalar
            result[key] = val.strip('"').strip("'")
        i += 1
    return result


TASK = _load_task()
TASK_NAME: str = TASK.get("name", "Prompt Chain Output")
TASK_INPUTS: list[dict[str, str]] = TASK.get("inputs", [])
TASK_INPUT_NAMES: list[str] = [inp["name"] for inp in TASK_INPUTS]
TASK_OUTPUT: dict[str, str] = TASK.get("output", {"name": "output", "description": "Generated output"})
TASK_JUDGE_ROLE: str = TASK.get("judge_role", "You are an expert evaluator.").strip()

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CASES_DIR = Path(__file__).parent / "test_cases"
LOCK_FILE = Path(__file__).parent / ".research.lock"
LOCK_STALE_SECONDS = 600  # 10 minutes
JUDGE_MODEL = os.environ.get("JUDGE_MODEL", "gpt-5-mini")
MAX_CONCURRENT = int(os.environ.get("EVAL_CONCURRENCY", "3"))
COST_LAMBDA = float(os.environ.get("COST_LAMBDA", "1.0"))

# ---------------------------------------------------------------------------
# Scoring rubric (loaded from criteria.md)
# ---------------------------------------------------------------------------

CRITERIA_FILE = Path(__file__).parent / "criteria.md"


def _load_criteria() -> tuple[list[str], str]:
    """Parse criteria.md -> (list of criterion names, judge system prompt).

    The judge role and input descriptions are read from task.yaml so the
    prompt adapts automatically when the task domain changes.
    """
    text = CRITERIA_FILE.read_text()
    names: list[str] = []
    descriptions: list[str] = []
    for line in text.splitlines():
        line = line.strip()
        if not line or not line[0].isdigit():
            continue
        rest = line.split(". ", 1)[1] if ". " in line else line
        if rest.startswith("**") and "**:" in rest:
            name = rest.split("**")[1]
            desc = rest.split("**: ", 1)[1] if "**: " in rest else rest
            names.append(name)
            descriptions.append(desc)

    criteria_block = "\n".join(
        f"{i+1}. {names[i]}: {descriptions[i]}" for i in range(len(names))
    )

    # Build input description lines from task.yaml
    input_lines = "\n".join(
        f"- {inp.get('description', inp['name'])}" for inp in TASK_INPUTS
    )
    output_desc = TASK_OUTPUT.get("description", "generated output")

    prompt = (
        f"{TASK_JUDGE_ROLE} You will be given:\n"
        f"{input_lines}\n"
        f"- The {output_desc}\n\n"
        "Score the output on each binary criterion below. Answer YES (1) or NO (0) for each.\n"
        "Be strict and objective.\n\n"
        f"Criteria:\n{criteria_block}\n\n"
        "Output ONLY valid JSON with no extra text:\n"
        '{"scores": [' + ",".join("0" for _ in range(len(names)))
        + f'], "total": {len(names)}' + "}"
    )
    return names, prompt


CRITERIA, JUDGE_SYSTEM = _load_criteria()


async def judge_output(inputs: dict[str, str], output: str) -> dict[str, int]:
    """Score a chain output using binary criteria. Returns per-criterion scores."""
    # Build judge prompt from task inputs dynamically
    input_sections = "\n\n".join(
        f"## {name.replace('_', ' ').title()}\n{value}"
        for name, value in inputs.items()
    )
    output_label = TASK_OUTPUT.get("name", "output").replace("_", " ").title()
    prompt = f"{input_sections}\n\n## Generated {output_label}\n{output}"
    response = await call_llm(
        messages=[
            {"role": "system", "content": JUDGE_SYSTEM},
            {"role": "user", "content": prompt},
        ],
        model=JUDGE_MODEL,
        temperature=0.0,
        max_tokens=512,
        reasoning_effort="low",
    )

    try:
        text = response.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
        raw: dict[str, Any] = json.loads(text)
        score_list: list[int] = raw["scores"]
        scores = {CRITERIA[i]: score_list[i] for i in range(len(CRITERIA))}
    except (json.JSONDecodeError, IndexError, KeyError) as exc:
        print(f"  WARNING: judge parse error ({type(exc).__name__}): {response[:200]}")
        scores = {c: 0 for c in CRITERIA}
    return scores


# ---------------------------------------------------------------------------
# Test case loading
# ---------------------------------------------------------------------------


def load_test_cases(max_cases: int | None = None) -> list[dict[str, Any]]:
    """Load test cases from test_cases/ directory."""
    cases: list[dict[str, Any]] = []
    case_files = sorted(CASES_DIR.glob("*.json"))
    if not case_files:
        print(f"ERROR: No test cases found in {CASES_DIR}/")
        sys.exit(1)
    for f in case_files:
        with open(f) as fh:
            case: dict[str, Any] = json.load(fh)
        case["_file"] = f.name
        # Validate that test case has all required input fields from task.yaml
        missing = [name for name in TASK_INPUT_NAMES if name not in case]
        if missing:
            print(f"ERROR: {f.name} missing required input fields: {missing}")
            print(f"  task.yaml expects: {TASK_INPUT_NAMES}")
            sys.exit(1)
        cases.append(case)
    if max_cases is not None:
        cases = cases[:max_cases]
    return cases


# ---------------------------------------------------------------------------
# Helpers for serializing LLMUsage
# ---------------------------------------------------------------------------


def _usage_to_dict(u: LLMUsage) -> dict[str, Any]:
    return {
        "input_tokens": u.input_tokens,
        "output_tokens": u.output_tokens,
        "calls": u.calls,
        "cost_usd": u.cost_usd,
        "duration_s": u.duration_s,
    }


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------


async def evaluate_case(
    case: dict[str, Any], case_idx: int, total: int
) -> dict[str, Any]:
    """Run chain + judge on a single test case."""
    import chain

    # Build inputs dict from task-defined field names
    inputs = {name: case[name] for name in TASK_INPUT_NAMES}
    label: str = case.get("label", case["_file"])

    print(f"  [{case_idx + 1}/{total}] {label} ... ", end="", flush=True)

    reset_usage()
    try:
        output = await chain.run_chain(inputs)
    except Exception as e:
        print(f"CHAIN ERROR: {e}")
        return {
            "label": label,
            "scores": {c: 0 for c in CRITERIA},
            "avg_score": 0.0,
            "chain_cost_usd": 0.0,
            "chain_usage": LLMUsage(),
            "judge_usage": LLMUsage(),
            "error": str(e),
            "output": "",
        }
    chain_usage_snapshot = LLMUsage(
        input_tokens=get_usage().input_tokens,
        output_tokens=get_usage().output_tokens,
        calls=get_usage().calls,
        cost_usd=get_usage().cost_usd,
        duration_s=get_usage().duration_s,
    )

    reset_usage()
    scores = await judge_output(inputs, output)
    judge_usage_snapshot = LLMUsage(
        input_tokens=get_usage().input_tokens,
        output_tokens=get_usage().output_tokens,
        calls=get_usage().calls,
        cost_usd=get_usage().cost_usd,
        duration_s=get_usage().duration_s,
    )

    total_score = sum(scores.values())
    score_10 = total_score / len(scores) * 10 if scores else 0.0
    passed = [c for c, v in scores.items() if v == 1]
    failed = [c for c, v in scores.items() if v == 0]
    print(
        f"{score_10:.1f}/10  "
        f"pass=[{','.join(passed)}]  "
        f"fail=[{','.join(failed)}]"
    )

    return {
        "label": label,
        "scores": scores,
        "avg_score": score_10,
        "chain_cost_usd": chain_usage_snapshot.cost_usd,
        "chain_usage": chain_usage_snapshot,
        "judge_usage": judge_usage_snapshot,
        "error": None,
        "output": output,
    }


async def run_evaluation(
    max_cases: int | None = None, verbose: bool = False
) -> dict[str, Any]:
    """Run full evaluation across all test cases. Persists to experiments/."""
    t_start = time.time()
    cases = load_test_cases(max_cases)
    print(f"Evaluating {len(cases)} test cases...")
    print()

    # Create run directory
    run_id = store.create_run()

    # Snapshot criteria, task definition, and chain code
    store.save_criteria_snapshot(run_id, CRITERIA_FILE.read_text())
    store.save_text_snapshot(run_id, "task.yaml", TASK_FILE.read_text())
    store.save_text_snapshot(run_id, "chain.py", (Path(__file__).parent / "chain.py").read_text())

    # Reload chain module to pick up edits since last import
    import chain

    importlib.reload(chain)

    # Run cases in parallel
    sem = asyncio.Semaphore(MAX_CONCURRENT)

    async def _run_with_semaphore(case: dict[str, Any], idx: int) -> dict[str, Any]:
        async with sem:
            return await evaluate_case(case, idx, len(cases))

    loop = asyncio.get_running_loop()
    tasks = []
    for i, case in enumerate(cases):
        ctx = contextvars.copy_context()
        t = loop.create_task(_run_with_semaphore(case, i), context=ctx)
        tasks.append(t)
    results = list(await asyncio.gather(*tasks))

    # Save per-case results
    for r in results:
        store.save_case_result(
            run_id=run_id,
            label=r["label"],
            output=r["output"],
            scores=r["scores"],
            score=r["avg_score"],
            chain_usage=_usage_to_dict(r["chain_usage"]),
            judge_usage=_usage_to_dict(r["judge_usage"]),
            error=r["error"],
        )

    # Aggregate
    total_chain_cost = 0.0
    total_chain_calls = 0
    total_chain_duration = 0.0
    total_judge_cost = 0.0
    all_scores: dict[str, list[int]] = {c: [] for c in CRITERIA}
    errors = 0

    for r in results:
        cu: LLMUsage = r["chain_usage"]
        total_chain_cost += cu.cost_usd
        total_chain_calls += cu.calls
        total_chain_duration += cu.duration_s

        ju: LLMUsage = r["judge_usage"]
        total_judge_cost += ju.cost_usd

        if r["error"]:
            errors += 1
        else:
            for k, v in r["scores"].items():
                all_scores[k].append(v)

    pass_rates = {k: sum(v) / len(v) if v else 0 for k, v in all_scores.items()}
    overall_avg = sum(pass_rates.values()) / len(pass_rates) * 10 if pass_rates else 0
    adjusted_score = overall_avg - COST_LAMBDA * total_chain_cost

    if verbose:
        print()
        for r in results:
            if not r["error"]:
                print(f"--- {r['label']} (avg={r['avg_score']:.1f}) ---")
                print(r["output"][:500])
                print("...\n")

    elapsed_seconds = time.time() - t_start
    summary: dict[str, Any] = {
        "score": overall_avg,
        "adjusted_score": adjusted_score,
        "cost_lambda": COST_LAMBDA,
        "chain_cost_usd": total_chain_cost,
        "judge_cost_usd": total_judge_cost,
        "total_cost_usd": total_chain_cost + total_judge_cost,
        "chain_calls": total_chain_calls,
        "chain_duration_s": total_chain_duration,
        "elapsed_seconds": elapsed_seconds,
        "num_cases": len(results),
        "num_errors": errors,
        "criteria_pass_rates": pass_rates,
    }
    store.save_summary(run_id, summary)

    # Stamp lock so --calibrate knows a research loop is active
    LOCK_FILE.write_text(time.strftime("%Y-%m-%dT%H:%M:%S"))

    # Attach transient fields for in-process use (not persisted)
    summary["run_id"] = run_id
    summary["results"] = results
    # Keep old keys for stdout printing
    summary["overall_avg"] = overall_avg
    summary["dimension_avgs"] = pass_rates

    return summary


# ---------------------------------------------------------------------------
# Human review (decoupled from live evaluation)
# ---------------------------------------------------------------------------


def run_review(run_id: str | None = None) -> None:
    """Interactively review stored outputs for a run.

    If run_id is None, reviews the most recent run.
    """
    if run_id is None:
        run_id = store.latest_run()
    if run_id is None:
        print("No experiment runs found. Run an evaluation first.")
        return

    try:
        meta = store.load_meta(run_id)
    except FileNotFoundError:
        print(f"Run '{run_id}' not found.")
        return

    case_labels = store.list_cases(run_id)
    if not case_labels:
        print(f"No cases found in run {run_id}.")
        return

    existing_reviews = store.load_reviews(run_id)
    approved = 0
    reviewed = 0

    print("\n" + "=" * 60)
    print(f"HUMAN REVIEW -- run {run_id}")
    print(f"  commit: {meta.get('commit', '?')}  branch: {meta.get('branch', '?')}")
    print(f"  status: {meta.get('status', '?')}")
    print("  y = good   n = bad   s = skip   q = quit review")
    print("=" * 60)

    for label in case_labels:
        if label in existing_reviews:
            prev = existing_reviews[label]
            print(f"\n  [{label}] already reviewed: {'approved' if prev['approved'] else 'rejected'} -- skipping")
            continue

        try:
            output_text, scores_data = store.load_case_result(run_id, label)
        except FileNotFoundError:
            continue

        scores = scores_data.get("scores", {})
        score_10 = scores_data.get("score", 0)
        error = scores_data.get("error")

        if error:
            print(f"\n  [{label}] had chain error: {error} -- skipping")
            continue

        failed = [c for c, v in scores.items() if v == 0]

        print(f"\n{'--' * 30}")
        print(f"Case: {label}  |  Judge: {score_10:.1f}/10  |  Failed: {failed or 'none'}")
        print(f"{'--' * 30}")
        if len(output_text) > 2200:
            print(output_text[:1500])
            print(f"\n  [...{len(output_text) - 2000} chars omitted...]\n")
            print(output_text[-500:])
        else:
            print(output_text)
        print(f"{'--' * 30}")

        while True:
            try:
                choice = input(f"  [{label}] (y)es / (n)o / (s)kip / (q)uit: ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                choice = "q"
            if choice in ("y", "n", "s", "q"):
                break
            print("  Invalid input. Use y/n/s/q.")

        if choice == "q":
            print("Review ended early.")
            break
        if choice == "s":
            continue

        rating = choice == "y"

        # Optional notes
        notes = ""
        with contextlib.suppress(EOFError, KeyboardInterrupt):
            notes = input("  Notes (Enter to skip): ").strip()

        store.save_review(run_id, label, approved=rating, notes=notes)
        reviewed += 1
        if rating:
            approved += 1

        # Flag mismatches
        if score_10 >= 7 and not rating:
            print(f"  FALSE POSITIVE: judge={score_10:.1f}/10 but human=reject")
        elif score_10 <= 4 and rating:
            print(f"  HIDDEN GEM: judge={score_10:.1f}/10 but human=approve")

    if reviewed == 0:
        print("No new ratings recorded.")
        return

    print(f"\nSaved {reviewed} reviews to experiments/{run_id}/reviews/")
    print(f"Approved: {approved}/{reviewed} ({approved / reviewed * 100:.0f}%)")


# ---------------------------------------------------------------------------
# Report across experiment runs
# ---------------------------------------------------------------------------


def run_report() -> None:
    """Print a summary table across all experiment runs."""
    runs = store.list_runs()
    if not runs:
        print("No experiment runs found.")
        return

    # Header
    print(f"{'run_id':<32s}  {'status':<8s}  {'adj_score':>9s}  {'score':>6s}  "
          f"{'cost':>7s}  {'cases':>5s}  {'reviews':>7s}  description")
    print("-" * 120)

    for run_id in runs:
        try:
            meta = store.load_meta(run_id)
        except (FileNotFoundError, json.JSONDecodeError):
            continue

        status = meta.get("status", "?")
        description = meta.get("description", "")

        try:
            summary = store.load_summary(run_id)
            adj = summary.get("adjusted_score", 0)
            score = summary.get("score", 0)
            cost = summary.get("chain_cost_usd", 0)
            n_cases = summary.get("num_cases", 0)
        except (FileNotFoundError, json.JSONDecodeError):
            adj = score = cost = 0.0
            n_cases = 0

        reviews = store.load_reviews(run_id)
        n_reviews = len(reviews)

        print(f"{run_id:<32s}  {status:<8s}  {adj:>9.4f}  {score:>6.4f}  "
              f"${cost:>6.4f}  {n_cases:>5d}  {n_reviews:>7d}  {description}")


# ---------------------------------------------------------------------------
# Criteria calibration
# ---------------------------------------------------------------------------

CALIBRATE_MIN_REVIEWS = int(os.environ.get("CALIBRATE_MIN_REVIEWS", "5"))
CALIBRATE_MODEL = os.environ.get("CALIBRATE_MODEL", "gpt-5-mini")

CALIBRATE_PROMPT = """\
You are calibrating binary evaluation criteria for a scoring system that evaluates: {task_name}.

An LLM judge scores outputs against these criteria, and a human separately rates \
each output as good (approved) or bad (rejected). We've found mismatches -- the \
criteria don't always predict human judgment.

## Current criteria.md
---
{criteria_md}
---

## Mismatch analysis
{analysis}

## Your task
Revise `criteria.md` to better align with human judgment. You may:
- Reword criteria that are too loose (cause false positives) or too strict (miss hidden gems)
- Remove criteria that don't correlate with human approval
- Add new criteria if the analysis reveals unmeasured qualities humans care about
- Adjust specificity (e.g., "at least 10 exchanges" might be wrong threshold)

Rules:
- Keep the same markdown format: numbered list, **bold_name**: description
- Criterion names must be valid Python identifiers (snake_case)
- Each criterion must be answerable as binary yes/no from the output text alone
- Update the header line to reflect the new count if it changed

Output ONLY the complete revised criteria.md content. No explanation, no fences."""


def _check_research_lock() -> bool:
    """Check if a research loop is active. Returns True if safe to proceed."""
    if not LOCK_FILE.exists():
        return True
    age = time.time() - LOCK_FILE.stat().st_mtime
    if age < LOCK_STALE_SECONDS:
        mins = int(age // 60)
        print(
            f"WARNING: A research loop was active {mins}m ago "
            f"(.research.lock modified {mins}m ago).\n"
            f"Changing criteria.md while the agent is running will cause "
            f"inconsistent scoring between iterations.\n"
        )
        while True:
            try:
                choice = input("Continue anyway? (y)es / (n)o: ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                choice = "n"
            if choice in ("y", "n"):
                break
        if choice == "n":
            print("Aborted. Stop the research loop first, then re-run --calibrate.")
            return False
    return True


def run_calibrate() -> None:
    """Analyze experiment reviews for judge-vs-human mismatches and propose criteria changes."""
    global CRITERIA, JUDGE_SYSTEM
    if not _check_research_lock():
        return

    reviews = store.load_all_reviews()
    if not reviews:
        print("No reviews found. Run --review first to collect human ratings.")
        return

    if len(reviews) < CALIBRATE_MIN_REVIEWS:
        print(
            f"Only {len(reviews)} reviews found, need at least {CALIBRATE_MIN_REVIEWS}. "
            f"Run more --review sessions first."
        )
        return

    approved_reviews = [r for r in reviews if r["approved"]]
    rejected_reviews = [r for r in reviews if not r["approved"]]

    print(f"Analyzing {len(reviews)} reviews ({len(approved_reviews)} approved, {len(rejected_reviews)} rejected)")
    print()

    analysis_lines: list[str] = []
    analysis_lines.append(f"Total reviews: {len(reviews)}")
    analysis_lines.append(f"Human approved: {len(approved_reviews)}, rejected: {len(rejected_reviews)}")
    analysis_lines.append("")

    analysis_lines.append("Per-criterion pass rates (approved vs rejected):")
    suspect_loose: list[str] = []
    suspect_strict: list[str] = []

    for criterion in CRITERIA:
        app_pass = sum(1 for r in approved_reviews if r.get("judge_scores", {}).get(criterion, 0) == 1)
        app_total = len(approved_reviews) or 1
        rej_pass = sum(1 for r in rejected_reviews if r.get("judge_scores", {}).get(criterion, 0) == 1)
        rej_total = len(rejected_reviews) or 1

        app_rate = app_pass / app_total
        rej_rate = rej_pass / rej_total

        status = ""
        if rej_rate > 0.5:
            status = " <- LOOSE (passes on rejected outputs)"
            suspect_loose.append(criterion)
        elif app_rate < 0.5:
            status = " <- STRICT (fails on approved outputs)"
            suspect_strict.append(criterion)

        line = f"  {criterion}: approved={app_rate:.0%}, rejected={rej_rate:.0%}{status}"
        analysis_lines.append(line)
        print(line)

    analysis_lines.append("")

    false_pos = [r for r in reviews if r.get("judge_score", 0) >= 7 and not r["approved"]]
    hidden_gems = [r for r in reviews if r.get("judge_score", 0) <= 4 and r["approved"]]

    if false_pos:
        analysis_lines.append(f"FALSE POSITIVES ({len(false_pos)} cases -- high judge score, human rejected):")
        for r in false_pos:
            failed = [c for c, v in r.get("judge_scores", {}).items() if v == 0]
            analysis_lines.append(f"  {r['label']}: score={r.get('judge_score', 0):.1f}/10, failed={failed}")
        analysis_lines.append("")

    if hidden_gems:
        analysis_lines.append(f"HIDDEN GEMS ({len(hidden_gems)} cases -- low judge score, human approved):")
        for r in hidden_gems:
            passed = [c for c, v in r.get("judge_scores", {}).items() if v == 1]
            analysis_lines.append(f"  {r['label']}: score={r.get('judge_score', 0):.1f}/10, passed={passed}")
        analysis_lines.append("")

    if suspect_loose:
        analysis_lines.append(f"SUSPECT LOOSE criteria (pass too often on rejected): {suspect_loose}")
    if suspect_strict:
        analysis_lines.append(f"SUSPECT STRICT criteria (fail too often on approved): {suspect_strict}")
    if not suspect_loose and not suspect_strict and not false_pos and not hidden_gems:
        print("\nNo significant mismatches found. Criteria appear well-calibrated.")
        return

    analysis_text = "\n".join(analysis_lines)
    current_criteria = CRITERIA_FILE.read_text()

    print("\nGenerating revised criteria...")
    revised = asyncio.run(_generate_calibration(current_criteria, analysis_text))
    if not revised:
        print("Failed to generate revision.")
        return

    print("\n" + "=" * 60)
    print("PROPOSED REVISED criteria.md")
    print("=" * 60)
    print(revised)
    print("=" * 60)

    while True:
        try:
            choice = input("\nApply this revision? (y)es / (n)o: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            choice = "n"
        if choice in ("y", "n"):
            break

    if choice == "y":
        CRITERIA_FILE.write_text(revised)
        CRITERIA, JUDGE_SYSTEM = _load_criteria()
        print(f"Updated {CRITERIA_FILE.name} with {len(CRITERIA)} criteria.")
    else:
        print("No changes applied.")


async def _generate_calibration(current_criteria: str, analysis: str) -> str | None:
    """Call LLM to propose revised criteria."""
    try:
        return await call_llm(
            messages=[
                {"role": "system", "content": CALIBRATE_PROMPT.format(
                    task_name=TASK_NAME,
                    criteria_md=current_criteria,
                    analysis=analysis,
                )},
                {"role": "user", "content": "Revise the criteria based on the mismatch analysis."},
            ],
            model=CALIBRATE_MODEL,
            reasoning_effort="high",
            max_tokens=4096,
        )
    except Exception as e:
        print(f"LLM call failed: {e}")
        return None


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description=f"Evaluate prompt chain for: {TASK_NAME}"
    )
    parser.add_argument(
        "--cases", type=int, default=None, help="Max test cases to run (default: all)"
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Print per-case output excerpts"
    )
    parser.add_argument(
        "--review", nargs="?", const="__latest__", default=None, metavar="RUN_ID",
        help="Review stored outputs (default: latest run)"
    )
    parser.add_argument(
        "--calibrate", action="store_true", help="Analyze reviews and propose criteria revisions"
    )
    parser.add_argument(
        "--report", action="store_true", help="Print summary table across all runs"
    )
    args = parser.parse_args()

    if args.calibrate:
        run_calibrate()
        return

    if args.report:
        run_report()
        return

    # --review dispatch: three modes
    #   --review RUN_ID         -> review that run, no evaluation
    #   --review (alone)        -> review latest run, no evaluation
    #   --review --cases N      -> evaluate first, then review the new run
    if args.review is not None and args.review != "__latest__":
        run_review(args.review)
        return

    if args.review == "__latest__" and args.cases is None:
        run_review(None)
        return

    # Run evaluation (optionally followed by --review of the new run)
    t0 = time.time()
    summary = asyncio.run(run_evaluation(max_cases=args.cases, verbose=args.verbose))
    elapsed = time.time() - t0

    print()
    print("---")
    print(f"run_id:           {summary['run_id']}")
    print(f"score:            {summary['overall_avg']:.4f}")
    print(f"adjusted_score:   {summary['adjusted_score']:.4f}")
    print(f"cost_lambda:      {COST_LAMBDA:.1f}")
    dim_avgs = summary["dimension_avgs"]
    max_key_len = max(len(c) for c in CRITERIA)
    for c in CRITERIA:
        pass_rate = dim_avgs[c]
        bar = "pass" if pass_rate >= 0.5 else "FAIL"
        print(f"  {c:<{max_key_len}}  {pass_rate:.0%}  {bar}")
    print(f"cases:            {summary['num_cases']} ({summary['num_errors']} errors)")
    print(f"chain_cost_usd:   {summary['chain_cost_usd']:.4f}")
    print(f"judge_cost_usd:   {summary['judge_cost_usd']:.4f}")
    print(f"total_cost_usd:   {summary['total_cost_usd']:.4f}")
    print(f"chain_calls:      {summary['chain_calls']}")
    print(f"elapsed_seconds:  {elapsed:.1f}")

    if args.review is not None:
        # --review alongside evaluation: review the just-completed run
        run_review(summary["run_id"])


if __name__ == "__main__":
    main()
