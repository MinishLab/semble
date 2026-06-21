from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import pytest

from benchmarks.swe import agent_run, utils
from benchmarks.swe.backends import Backend
from benchmarks.swe.utils import WITH_SEMBLE, WITHOUT_SEMBLE, ParsedRun, RunResult, TaskResult


@dataclass
class _FakeTask:
    instance_id: str = "pytest-dev__pytest-123"
    repo: str = "pytest-dev/pytest"
    base_commit: str = "abc123"
    problem_statement: str = "Fix the bug"
    gold_files: list[str] = None  # type: ignore[assignment]
    short_label: str = "Fix the bug"

    def __post_init__(self) -> None:
        if self.gold_files is None:
            self.gold_files = ["src/app.py"]


class _ScriptedBackend(Backend):
    name = "fake"
    default_model = "model-a"

    def __init__(self, outcomes: list[RunResult | BaseException]) -> None:
        super().__init__(model=self.default_model)
        self._outcomes = outcomes
        self.calls = 0

    def label(self) -> str:
        return f"{self.name}/{self.model}"

    def _run_once(self, prompt: str, repo: Path, *, with_semble: bool) -> tuple[ParsedRun, str]:
        del prompt, repo, with_semble
        raise AssertionError("_run_once should not be called in this test backend")

    def run(self, prompt: str, repo: Path, commit: str, *, with_semble: bool) -> RunResult:
        del prompt, repo, commit, with_semble
        self.calls += 1
        outcome = self._outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


def _configure_paths(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    repos_dir = tmp_path / "repos"
    results_dir = tmp_path / "results"
    monkeypatch.setattr(agent_run, "REPOS_DIR", repos_dir)
    monkeypatch.setattr(agent_run, "RESULTS_DIR", results_dir)
    monkeypatch.setattr(utils, "REPOS_DIR", repos_dir)
    monkeypatch.setattr(utils, "RESULTS_DIR", results_dir)


def _load_results(results_dir: Path) -> list[dict]:
    return json.loads((results_dir / "swe_agent.json").read_text())


def test_run_resume_continues_after_interrupted_variant(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """A resumed run should skip the already-persisted variant and continue with the missing one."""
    _configure_paths(monkeypatch, tmp_path)
    task = _FakeTask()

    monkeypatch.setattr(agent_run, "resolve_tasks", lambda *args, **kwargs: [task])
    monkeypatch.setattr(agent_run, "clone_at_commit", lambda *args, **kwargs: None)

    first_variant = RunResult(
        variant="",
        backend="fake",
        model="model-a",
        patch="diff --git a/src/app.py b/src/app.py",
        touched_files=["src/app.py"],
    )
    first_backend = _ScriptedBackend([first_variant, KeyboardInterrupt()])

    with pytest.raises(KeyboardInterrupt):
        agent_run.run(first_backend, n_tasks=1, resume=False)

    results = _load_results(tmp_path / "results")
    assert len(results) == 1
    assert [r["variant"] for r in results[0]["results"]] == [WITH_SEMBLE]

    second_variant = RunResult(
        variant="",
        backend="fake",
        model="model-a",
        patch="diff --git a/src/app.py b/src/app.py\n+fix",
        touched_files=["src/app.py"],
    )
    second_backend = _ScriptedBackend([second_variant])
    agent_run.run(second_backend, n_tasks=1, resume=True)

    assert second_backend.calls == 1
    results = _load_results(tmp_path / "results")
    saved_variants = {r["variant"] for r in results[0]["results"]}
    assert saved_variants == {WITH_SEMBLE, WITHOUT_SEMBLE}

    without_predictions = (
        (tmp_path / "results" / "predictions_without_semble_fake-model-a.jsonl").read_text().strip().splitlines()
    )
    assert len(without_predictions) == 1


def test_save_outputs_overwrites_existing_backend_model_variant(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Saving a rerun should replace the same backend/model/variant slot instead of duplicating it."""
    _configure_paths(monkeypatch, tmp_path)
    existing = [
        TaskResult(
            instance_id="pytest-dev__pytest-123",
            gold_files=["src/app.py"],
            results=[
                RunResult(
                    variant=WITH_SEMBLE,
                    backend="fake",
                    model="model-a",
                    patch="old patch",
                    touched_files=["src/app.py"],
                )
            ],
        )
    ]
    (tmp_path / "results").mkdir(parents=True, exist_ok=True)
    (tmp_path / "results" / "swe_agent.json").write_text(json.dumps([asdict(t) for t in existing]))

    updated = TaskResult(
        instance_id="pytest-dev__pytest-123",
        gold_files=["src/app.py"],
        results=[
            RunResult(
                variant=WITH_SEMBLE,
                backend="fake",
                model="model-a",
                patch="new patch",
                touched_files=["src/app.py"],
            )
        ],
    )

    agent_run._save_outputs([updated], "fake", "model-a", "fake/model-a")

    results = _load_results(tmp_path / "results")
    assert len(results) == 1
    assert len(results[0]["results"]) == 1
    assert results[0]["results"][0]["patch"] == "new patch"
