"""
Experiment storage layer.

All reads and writes to the experiments/ directory go through this module.
This keeps storage logic out of evaluate.py and provides a clean API for
all consumers (evaluation, review, calibration, analysis).

Directory layout per run:
    experiments/<run_id>/
        meta.json           -- run identity and status
        task.yaml           -- snapshot of task definition used
        criteria.md         -- snapshot of criteria used
        chain.py            -- snapshot of chain code used
        summary.json        -- aggregate scores, costs, timings
        cases/
            <label>/
                output.md   -- generated output
                scores.json -- per-criterion judge scores + usage
        reviews/
            <label>.json    -- human review (added asynchronously)
"""

from __future__ import annotations

import contextlib
import json
import os
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

EXPERIMENTS_DIR = Path(__file__).parent / "experiments"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_atomic(path: Path, content: str) -> None:
    """Write a text file atomically (write to temp, then rename)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(content)
        os.replace(tmp, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp)
        raise


def _write_json(path: Path, data: dict[str, Any]) -> None:
    """Write JSON atomically (write to temp, then rename)."""
    _write_atomic(path, json.dumps(data, indent=2, default=str) + "\n")


def _read_json(path: Path) -> dict[str, Any]:
    """Read a JSON file."""
    with open(path) as f:
        return json.load(f)  # type: ignore[no-any-return]


def _git_info() -> tuple[str, str]:
    """Return (short_commit, branch) from git. Falls back to ('unknown', 'unknown')."""
    commit = "unknown"
    branch = "unknown"
    try:
        cp = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True,
        )
        commit = cp.stdout.strip() or "unknown"
    except Exception:
        pass
    try:
        cp = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True, text=True,
        )
        branch = cp.stdout.strip() or "unknown"
    except Exception:
        pass
    return commit, branch


# ---------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------


def create_run(commit: str | None = None, branch: str | None = None) -> str:
    """Create a new run directory. Returns the run_id."""
    if commit is None or branch is None:
        git_commit, git_branch = _git_info()
        commit = commit or git_commit
        branch = branch or git_branch

    timestamp = time.strftime("%Y%m%dT%H%M%S")
    run_id = f"{timestamp}_{commit}"
    run_dir = EXPERIMENTS_DIR / run_id

    # Avoid collision if same timestamp+commit (add suffix)
    if run_dir.exists():
        for i in range(1, 100):
            candidate = f"{run_id}_{i}"
            if not (EXPERIMENTS_DIR / candidate).exists():
                run_id = candidate
                run_dir = EXPERIMENTS_DIR / run_id
                break
        else:
            raise RuntimeError(f"Too many runs with same timestamp+commit: {run_id}")

    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "cases").mkdir(exist_ok=True)
    (run_dir / "reviews").mkdir(exist_ok=True)

    meta = {
        "run_id": run_id,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "commit": commit,
        "branch": branch,
        "status": "pending",
        "description": "",
    }
    _write_json(run_dir / "meta.json", meta)
    return run_id


def save_text_snapshot(run_id: str, filename: str, text: str) -> None:
    """Copy a text file into the run directory (for criteria.md, task.yaml, etc.)."""
    _write_atomic(EXPERIMENTS_DIR / run_id / filename, text)


def save_criteria_snapshot(run_id: str, criteria_text: str) -> None:
    """Copy current criteria.md content into the run directory."""
    save_text_snapshot(run_id, "criteria.md", criteria_text)


def save_case_result(
    run_id: str,
    label: str,
    output: str,
    scores: dict[str, int],
    score: float,
    chain_usage: dict[str, Any],
    judge_usage: dict[str, Any],
    error: str | None,
) -> None:
    """Write output.md and scores.json for one case."""
    case_dir = EXPERIMENTS_DIR / run_id / "cases" / label
    case_dir.mkdir(parents=True, exist_ok=True)

    # Write output
    _write_atomic(case_dir / "output.md", output or "")

    # Write scores
    _write_json(case_dir / "scores.json", {
        "label": label,
        "score": score,
        "scores": scores,
        "chain_usage": chain_usage,
        "judge_usage": judge_usage,
        "error": error,
    })


def save_summary(run_id: str, summary: dict[str, Any]) -> None:
    """Write summary.json for the run."""
    _write_json(EXPERIMENTS_DIR / run_id / "summary.json", summary)


def update_status(run_id: str, status: str, description: str = "") -> None:
    """Update meta.json with final status and description."""
    meta_path = EXPERIMENTS_DIR / run_id / "meta.json"
    meta = _read_json(meta_path)
    meta["status"] = status
    if description:
        meta["description"] = description
    _write_json(meta_path, meta)


# ---------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------


def list_runs() -> list[str]:
    """List all run IDs, sorted chronologically (oldest first)."""
    if not EXPERIMENTS_DIR.exists():
        return []
    runs = []
    for d in EXPERIMENTS_DIR.iterdir():
        if d.is_dir() and (d / "meta.json").exists():
            runs.append(d.name)
    return sorted(runs)


def load_meta(run_id: str) -> dict[str, Any]:
    """Load meta.json for a run."""
    return _read_json(EXPERIMENTS_DIR / run_id / "meta.json")


def load_summary(run_id: str) -> dict[str, Any]:
    """Load summary.json for a run."""
    return _read_json(EXPERIMENTS_DIR / run_id / "summary.json")


def load_case_result(run_id: str, label: str) -> tuple[str, dict[str, Any]]:
    """Load (output_text, scores_dict) for a case."""
    case_dir = EXPERIMENTS_DIR / run_id / "cases" / label
    output_text = (case_dir / "output.md").read_text()
    scores = _read_json(case_dir / "scores.json")
    return output_text, scores


def list_cases(run_id: str) -> list[str]:
    """List case labels in a run."""
    cases_dir = EXPERIMENTS_DIR / run_id / "cases"
    if not cases_dir.exists():
        return []
    return sorted(d.name for d in cases_dir.iterdir() if d.is_dir())


def latest_run() -> str | None:
    """Return the most recent run_id, or None if no runs exist."""
    runs = list_runs()
    return runs[-1] if runs else None


# ---------------------------------------------------------------------------
# Reviews
# ---------------------------------------------------------------------------


def save_review(
    run_id: str,
    label: str,
    approved: bool,
    notes: str = "",
) -> None:
    """Write a human review for one case in a run."""
    reviews_dir = EXPERIMENTS_DIR / run_id / "reviews"
    reviews_dir.mkdir(parents=True, exist_ok=True)
    _write_json(reviews_dir / f"{label}.json", {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "approved": approved,
        "notes": notes,
    })


def load_reviews(run_id: str) -> dict[str, dict[str, Any]]:
    """Load all reviews for a run. Returns {label: review_dict}."""
    reviews_dir = EXPERIMENTS_DIR / run_id / "reviews"
    if not reviews_dir.exists():
        return {}
    result: dict[str, dict[str, Any]] = {}
    for f in reviews_dir.iterdir():
        if f.suffix == ".json":
            result[f.stem] = _read_json(f)
    return result


def load_all_reviews() -> list[dict[str, Any]]:
    """Load all reviews across all runs (for calibration).

    Returns a list of dicts, each containing the review data plus
    run_id, label, and the judge scores from scores.json.
    """
    all_reviews: list[dict[str, Any]] = []
    for run_id in list_runs():
        reviews = load_reviews(run_id)
        for label, review in reviews.items():
            entry: dict[str, Any] = {
                "run_id": run_id,
                "label": label,
                **review,
            }
            # Attach judge scores if available
            try:
                _, scores_data = load_case_result(run_id, label)
                entry["judge_score"] = scores_data.get("score", 0)
                entry["judge_scores"] = scores_data.get("scores", {})
            except (FileNotFoundError, json.JSONDecodeError):
                pass
            all_reviews.append(entry)
    return all_reviews


def list_unreviewed_runs() -> list[str]:
    """List runs that have cases without reviews."""
    unreviewed: list[str] = []
    for run_id in list_runs():
        cases = list_cases(run_id)
        reviews = load_reviews(run_id)
        if cases and len(reviews) < len(cases):
            unreviewed.append(run_id)
    return unreviewed
