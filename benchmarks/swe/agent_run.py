from __future__ import annotations

import argparse
import json
import subprocess
import time
from dataclasses import asdict
from pathlib import Path

from swebench.harness.constants import KEY_INSTANCE_ID, KEY_MODEL, KEY_PREDICTION

from benchmarks.swe.backends import _BACKENDS, Backend, ClaudeBackend, CodexBackend, OpencodeBackend
from benchmarks.swe.tasks import SWETask, resolve_tasks
from benchmarks.swe.utils import (
    DEFAULT_REPO,
    DEFAULT_SEED,
    PROBLEM_MAX_CHARS,
    PROMPT_BASE,
    REPOS_DIR,
    RESULTS_DIR,
    TASK_SLEEP,
    WITH_SEMBLE,
    WITHOUT_SEMBLE,
    RunResult,
    TaskResult,
    agent_results_path,
    bootstrap_ci,
    clone_at_commit,
    is_semble_tool_call,
    prediction_path,
    variant_name,
)


def _hits_gold(touched: list[str], gold: list[str]) -> bool:
    """Exact-path overlap with at least one gold file (no basename/suffix matching)."""
    return bool(set(touched) & set(gold))


def _completed_variants(backend: Backend, experiment: str | None = None) -> dict[str, set[str]]:
    """Return ``{instance_id: {variant, ...}}`` for successful, non-empty results from this backend+model."""
    out: dict[str, set[str]] = {}
    path = agent_results_path(experiment)
    if not path.exists():
        return out
    for entry in json.loads(path.read_text()):
        iid = entry["instance_id"]
        for r in entry["results"]:
            if (
                r.get("backend") == backend.name
                and r.get("model") == backend.model
                and not r.get("error")
                and r.get("patch")
            ):
                out.setdefault(iid, set()).add(r["variant"])
    return out


def _print_run_result(r: RunResult) -> None:
    if r.error:
        print(f"    ERROR: {r.error}")
        return
    semble_calls = sum(1 for t in r.tool_calls if is_semble_tool_call(t))
    bypass_note = "  [BYPASS]" if r.bypass else ""
    print(
        f"    turns={r.num_turns}  cost=${r.cost_usd:.3f}  out={r.output_tokens}  "
        f"semble={semble_calls}  gold_hit={r.gold_hit}{bypass_note}"
    )


def _run_variants(
    backend: Backend,
    task: SWETask,
    dest: Path,
    done_variants: set[str],
    *,
    experiment: str | None,
    with_semble_only: bool,
) -> list[RunResult]:
    """Run the with/without-semble variants for one task against an already-cloned repo."""
    results: list[RunResult] = []
    variants_to_run = [True] if with_semble_only else [True, False]
    for with_semble in variants_to_run:
        variant = variant_name(with_semble, experiment)
        run_label = "WITH semble   " if with_semble else "WITHOUT semble"
        if variant in done_variants:
            print(f"  [{run_label}] skipped")
            continue
        prompt = PROMPT_BASE.format(repo=dest, problem=task.problem_statement[:PROBLEM_MAX_CHARS])

        print(f"  [{run_label}] running...", flush=True)
        r = backend.run(prompt, dest, task.base_commit, with_semble=with_semble)
        r.variant = variant
        r.gold_hit = _hits_gold(r.touched_files, task.gold_files)
        _print_run_result(r)
        results.append(r)
    return results


def run(
    backend: Backend,
    n_tasks: int = 5,
    repo: str = DEFAULT_REPO,
    instance_ids: list[str] | None = None,
    resume: bool = False,
    experiment: str | None = None,
    with_semble_only: bool = False,
    seed: int = DEFAULT_SEED,
) -> None:
    """Run *backend* over SWE-bench Lite tasks, with and without semble, and save results."""
    REPOS_DIR.mkdir(exist_ok=True)
    RESULTS_DIR.mkdir(exist_ok=True)

    print(f"Backend: {backend.label()}")
    tasks = resolve_tasks(n_tasks, repo, instance_ids, seed)
    if not tasks:
        print(f"No tasks for repo={repo!r}")
        return
    print(f"Got {len(tasks)} tasks.\n")

    done = _completed_variants(backend, experiment) if resume else {}
    if done:
        skippable = sum(1 for iid, vs in done.items() if WITH_SEMBLE in vs and WITHOUT_SEMBLE in vs)
        print(f"Resume mode: {skippable} tasks already complete for {backend.label()}\n")

    all_rows: list[TaskResult] = []

    for task_i, task in enumerate(tasks):
        if not task.gold_files:
            continue

        iid = task.instance_id
        dest = REPOS_DIR / f"{task.repo.replace('/', '_')}_{task.base_commit}"

        done_variants = done.get(iid, set())
        if WITH_SEMBLE in done_variants and WITHOUT_SEMBLE in done_variants:
            print(f"[{task_i + 1}/{len(tasks)}] {iid}  (skipped — already done)")
            continue

        print(f"[{task_i + 1}/{len(tasks)}] {iid}")
        print(f"  gold:  {task.gold_files}")
        print(f"  issue: {task.short_label!r}")
        print("  clone...", end=" ", flush=True)
        try:
            clone_at_commit(task.repo, task.base_commit, dest)
            print("ok")
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
            print(f"FAILED: {exc}")
            continue

        results = _run_variants(
            backend,
            task,
            dest,
            done_variants,
            experiment=experiment,
            with_semble_only=with_semble_only,
        )
        all_rows.append(TaskResult(instance_id=iid, gold_files=task.gold_files, results=results))

        if task_i < len(tasks) - 1:
            print(f"  (sleeping {TASK_SLEEP}s...)\n")
            time.sleep(TASK_SLEEP)

    _print_summary(all_rows, backend.label(), experiment=experiment)
    _save_outputs(all_rows, backend.label(), experiment=experiment)


def _print_summary(rows: list[TaskResult], model_label: str, experiment: str | None = None) -> None:
    with_variant = variant_name(True, experiment)
    with_r = [r for row in rows for r in row.results if r.variant == with_variant and not r.error]
    without_r_all = [r for row in rows for r in row.results if r.variant == WITHOUT_SEMBLE and not r.error]
    without_r = [r for r in without_r_all if not r.bypass]
    n_bypass = len(without_r_all) - len(without_r)

    def avg(items: list[RunResult], key: str) -> float:
        vals = [getattr(r, key) for r in items]
        return sum(vals) / len(vals) if vals else 0.0

    def avg_semble_calls(items: list[RunResult]) -> float:
        vals = [sum(1 for t in r.tool_calls if is_semble_tool_call(t)) for r in items]
        return sum(vals) / len(vals) if vals else 0.0

    n = len(rows)
    exp_label = f" [{experiment}]" if experiment else ""
    print(f"\n{'=' * 60}")
    print(f"Summary ({n} tasks, {model_label}{exp_label})")
    if n_bypass:
        print(f"  ({n_bypass} without-semble runs excluded below — semble accessed via [SEMBLE_BYPASS])")
    print(f"  {'Metric':<25}  {'With Semble':>12}  {'Without':>9}")
    print(f"  {'-' * 25}  {'-' * 12}  {'-' * 9}")
    if with_r and without_r:
        cost_w_ci = bootstrap_ci([r.cost_usd for r in with_r])
        cost_wo_ci = bootstrap_ci([r.cost_usd for r in without_r])
        print(f"  {'avg cost (USD)':<25}  {avg(with_r, 'cost_usd'):>12.3f}  {avg(without_r, 'cost_usd'):>9.3f}")
        print(f"  {'  95% CI':<25}  [{cost_w_ci[0]:.3f},{cost_w_ci[1]:.3f}]  [{cost_wo_ci[0]:.3f},{cost_wo_ci[1]:.3f}]")
        print(
            f"  {'avg output tokens':<25}  "
            f"{avg(with_r, 'output_tokens'):>12.0f}  {avg(without_r, 'output_tokens'):>9.0f}"
        )
        print(f"  {'avg turns':<25}  {avg(with_r, 'num_turns'):>12.1f}  {avg(without_r, 'num_turns'):>9.1f}")
        print(f"  {'avg semble calls':<25}  {avg_semble_calls(with_r):>12.1f}  {avg_semble_calls(without_r):>9.1f}")
        ws_hit = sum(1 for r in with_r if r.gold_hit) / len(with_r)
        wo_hit = sum(1 for r in without_r if r.gold_hit) / len(without_r)
        print(f"  {'gold file hit rate':<25}  {ws_hit:>12.0%}  {wo_hit:>9.0%}")
        print(f"\n  n={len(with_r)} with-semble / n={len(without_r)} without-semble runs.")
        if min(len(with_r), len(without_r)) < 30:
            print("  Sample too small for any of the above to be statistically meaningful —")
            print("  treat as directional only. Use evaluate.py's resolve rate + McNemar test")
            print("  for the metric that actually matters, on >=30+ paired tasks.")
    elif with_r:
        print(f"  {'avg cost (USD)':<25}  {avg(with_r, 'cost_usd'):>12.3f}")
        print(f"  {'avg output tokens':<25}  {avg(with_r, 'output_tokens'):>12.0f}")
        print(f"  {'avg semble calls':<25}  {avg_semble_calls(with_r):>12.1f}")
        ws_hit = sum(1 for r in with_r if r.gold_hit) / len(with_r)
        print(f"  {'gold file hit rate':<25}  {ws_hit:>12.0%}")
    errors_w = sum(1 for row in rows for r in row.results if r.variant == with_variant and r.error)
    errors_wo = sum(1 for row in rows for r in row.results if r.variant == WITHOUT_SEMBLE and r.error)
    if errors_w or errors_wo:
        print(f"  {'errors':<25}  {errors_w:>12}  {errors_wo:>9}")


def _save_outputs(rows: list[TaskResult], model_label: str, experiment: str | None = None) -> None:
    out = agent_results_path(experiment)
    existing: dict[str, TaskResult] = {}
    if out.exists():
        for entry in json.loads(out.read_text()):
            tr = TaskResult.from_dict(entry)
            existing[tr.instance_id] = tr
    for row in rows:
        iid = row.instance_id
        if iid in existing:
            result_map = {(r.backend, r.model, r.variant): r for r in existing[iid].results}
            for new_r in row.results:
                key = (new_r.backend, new_r.model, new_r.variant)
                result_map[key] = new_r
            existing[iid].results = list(result_map.values())
        else:
            existing[iid] = row
    merged = list(existing.values())
    out.write_text(json.dumps([asdict(t) for t in merged], indent=2))
    print(f"\nFull results -> {out}  ({len(merged)} instances)")

    model_slug = model_label.replace("/", "-")
    for variant in (WITH_SEMBLE, WITHOUT_SEMBLE):
        result_variant = variant_name(variant == WITH_SEMBLE, experiment)
        predictions = [
            {
                KEY_INSTANCE_ID: t.instance_id,
                KEY_PREDICTION: r.patch,
                KEY_MODEL: f"{model_slug}-{variant}",
            }
            for t in merged
            for r in t.results
            if r.variant == result_variant and not r.error and r.patch and not r.bypass
        ]
        path = prediction_path(with_semble=variant == WITH_SEMBLE, experiment=experiment)
        path.write_text("\n".join(json.dumps(p) for p in predictions) + "\n")
        print(f"Predictions ({variant}): {path}  ({len(predictions)} patches)")

    ids = sorted(t.instance_id for t in merged)
    print("\nRun evaluation:")
    eval_cmd = "uv run python -m benchmarks.swe.evaluate"
    if experiment:
        eval_cmd += f" --experiment {experiment}"
    print(f"  {eval_cmd} --instance-ids {' '.join(ids[:5])}")
    if len(ids) > 5:
        print(f"  # (and {len(ids) - 5} more)")


def main() -> None:
    """Parse CLI arguments and run the SWE-bench agent benchmark."""
    p = argparse.ArgumentParser()
    p.add_argument("--backend", default="claude", choices=list(_BACKENDS))
    p.add_argument(
        "--model",
        default=None,
        help=(
            "Model override (defaults vary by backend: "
            f"claude={ClaudeBackend.default_model}, "
            f"codex={CodexBackend.default_model}, "
            f"opencode={OpencodeBackend.default_model})"
        ),
    )
    p.add_argument("--tasks", type=int, default=5, help="Number of tasks (ignored if --instance-ids given)")
    p.add_argument("--instance-ids", nargs="*", help="Specific instance IDs to run (default: first N from dataset)")
    p.add_argument(
        "--repo",
        default=DEFAULT_REPO,
        help="Repo name, comma-separated repo names, or 'all' to sample across the whole dataset",
    )
    p.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Seed for random task sampling")
    p.add_argument(
        "--resume", action="store_true", help="Skip tasks already successfully completed for this backend+model"
    )
    p.add_argument(
        "--experiment",
        default=None,
        help="Experiment name: saves to swe_agent_{NAME}.json and tags the with-semble variant",
    )
    p.add_argument(
        "--with-semble-only", action="store_true", help="Only run the WITH semble variant (skip WITHOUT semble)"
    )
    p.add_argument(
        "--local-semble",
        action="store_true",
        help="Use local branch semble via 'uv run semble' instead of installed PyPI version",
    )
    args = p.parse_args()

    backend_cls = _BACKENDS[args.backend]
    backend = backend_cls(model=args.model, local_semble=args.local_semble)
    run(
        backend,
        args.tasks,
        args.repo,
        instance_ids=args.instance_ids or None,
        resume=args.resume,
        experiment=args.experiment,
        with_semble_only=args.with_semble_only,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
