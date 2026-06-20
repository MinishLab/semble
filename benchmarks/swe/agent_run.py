from __future__ import annotations

import argparse
import json
import random
import subprocess
import time
import urllib.request
from dataclasses import asdict
from pathlib import Path

from benchmarks.swe.backends import _BACKENDS, Backend, ClaudeBackend, CodexBackend, OpencodeBackend, RunResult
from benchmarks.swe.backends.base import is_semble_tool_call
from benchmarks.swe.gitutils import changed_files, clone_at_commit
from benchmarks.swe.stats import bootstrap_ci

REPOS_DIR = Path(__file__).parent / "repos"
RESULTS_DIR = Path(__file__).parent / "results"
_HF_CHUNK = (
    "https://datasets-server.huggingface.co/rows"
    "?dataset=princeton-nlp%2FSWE-bench_Lite&config=default&split=test"
    "&offset={offset}&length=100"
)
_DEFAULT_REPO = "pytest-dev/pytest"
_DEFAULT_SEED = 42
_TASK_SLEEP = 30  # seconds between tasks (rate-limit buffer)
_PROBLEM_MAX_CHARS = 8000
_HF_CACHE = RESULTS_DIR / "swe_bench_lite_tasks.json"

_PROMPT_BASE = """\
You are a software engineer. Fix the following GitHub issue in the repository at {repo}.

{problem}

Instructions:
- Explore the repository to understand the relevant code
- Make the minimal change needed to fix the issue
- Do NOT run the test suite
- Do NOT add new test files
"""


def _fetch_tasks(
    n: int, repo: str = _DEFAULT_REPO, seed: int = _DEFAULT_SEED, refresh_cache: bool = False
) -> list[dict]:
    """Fetch SWE-bench Lite tasks, seeded-randomly sampled.

    *repo* accepts a single repo, a comma-separated list, or ``"all"``.
    """
    if refresh_cache and _HF_CACHE.exists():
        _HF_CACHE.unlink()
    if _HF_CACHE.exists():
        all_rows = json.loads(_HF_CACHE.read_text())
    else:
        all_rows = []
        for offset in range(0, 300, 100):
            with urllib.request.urlopen(_HF_CHUNK.format(offset=offset), timeout=30) as r:
                all_rows += [row["row"] for row in json.loads(r.read())["rows"]]
        RESULTS_DIR.mkdir(exist_ok=True)
        _HF_CACHE.write_text(json.dumps(all_rows))

    if repo == "all":
        pool = list(all_rows)
    else:
        repos = {r.strip() for r in repo.split(",")}
        pool = [r for r in all_rows if r["repo"] in repos]

    random.Random(seed).shuffle(pool)
    return pool[:n]


def _short_label(problem: str) -> str:
    """First line of the problem statement, for console progress output only (not fed to the agent)."""
    for line in problem.splitlines():
        if line.strip():
            return line.strip()[:200]
    return problem[:200]


def _hits_gold(touched: list[str], gold: list[str]) -> bool:
    """Exact-path overlap with at least one gold file (no basename/suffix matching)."""
    return bool(set(touched) & set(gold))


def _completed_variants(backend: Backend) -> dict[str, set[str]]:
    """Return ``{instance_id: {variant, ...}}`` for successful, non-empty results from this backend+model."""
    out: dict[str, set[str]] = {}
    path = RESULTS_DIR / "swe_agent.json"
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


def _resolve_tasks(
    n_tasks: int, repo: str, instance_ids: list[str] | None, seed: int, refresh_cache: bool
) -> list[dict]:
    """Fetch either a specific set of instance IDs or a fresh random sample, depending on what's requested."""
    if instance_ids:
        print(f"Fetching tasks for {len(instance_ids)} specific instance IDs...")
        tasks = _fetch_tasks(300, "all", seed=seed, refresh_cache=refresh_cache)
        tasks = [t for t in tasks if t["instance_id"] in set(instance_ids)]
        missing = set(instance_ids) - {t["instance_id"] for t in tasks}
        if missing:
            print(f"WARNING: {len(missing)} IDs not found: {missing}")
        return tasks
    print(f"Fetching {n_tasks} randomly sampled tasks from SWE-bench Lite (repo={repo}, seed={seed})...")
    return _fetch_tasks(n_tasks, repo, seed=seed, refresh_cache=refresh_cache)


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
    dest: Path,
    commit: str,
    problem: str,
    gold: list[str],
    done_variants: set[str],
    *,
    experiment: str | None,
    with_semble_only: bool,
) -> list[RunResult]:
    """Run the with/without-semble variants for one task against an already-cloned repo."""
    results: list[RunResult] = []
    variants_to_run = [True] if with_semble_only else [True, False]
    for with_semble in variants_to_run:
        base_variant = "with_semble" if with_semble else "without_semble"
        variant = f"{base_variant}_{experiment}" if experiment and with_semble else base_variant
        run_label = "WITH semble   " if with_semble else "WITHOUT semble"
        if variant in done_variants:
            print(f"  [{run_label}] skipped")
            continue
        prompt = _PROMPT_BASE.format(repo=dest, problem=problem[:_PROBLEM_MAX_CHARS])

        print(f"  [{run_label}] running...", flush=True)
        r = backend.run(prompt, dest, commit, with_semble=with_semble)
        r.variant = variant
        r.gold_hit = _hits_gold(r.touched_files, gold)
        _print_run_result(r)
        results.append(r)
    return results


def run(
    backend: Backend,
    n_tasks: int = 5,
    repo: str = _DEFAULT_REPO,
    instance_ids: list[str] | None = None,
    resume: bool = False,
    experiment: str | None = None,
    with_semble_only: bool = False,
    seed: int = _DEFAULT_SEED,
    refresh_cache: bool = False,
) -> None:
    """Run *backend* over SWE-bench Lite tasks, with and without semble, and save results."""
    REPOS_DIR.mkdir(exist_ok=True)
    RESULTS_DIR.mkdir(exist_ok=True)

    print(f"Backend: {backend.label()}")
    tasks = _resolve_tasks(n_tasks, repo, instance_ids, seed, refresh_cache)
    if not tasks:
        print(f"No tasks for repo={repo!r}")
        return
    print(f"Got {len(tasks)} tasks.\n")

    done = _completed_variants(backend) if resume else {}
    if done:
        skippable = sum(1 for iid, vs in done.items() if "with_semble" in vs and "without_semble" in vs)
        print(f"Resume mode: {skippable} tasks already complete for {backend.label()}\n")

    all_rows: list[dict] = []

    for task_i, task in enumerate(tasks):
        iid = task["instance_id"]
        task_repo = task["repo"]
        commit = task["base_commit"]
        problem = task["problem_statement"]
        gold = changed_files(task["patch"])
        if not gold:
            continue

        dest = REPOS_DIR / f"{task_repo.replace('/', '_')}_{commit}"

        done_variants = done.get(iid, set())
        if "with_semble" in done_variants and "without_semble" in done_variants:
            print(f"[{task_i + 1}/{len(tasks)}] {iid}  (skipped — already done)")
            continue

        print(f"[{task_i + 1}/{len(tasks)}] {iid}")
        print(f"  gold:  {gold}")
        print(f"  issue: {_short_label(problem)!r}")
        print("  clone...", end=" ", flush=True)
        try:
            clone_at_commit(task_repo, commit, dest)
            print("ok")
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
            print(f"FAILED: {exc}")
            continue

        results = _run_variants(
            backend,
            dest,
            commit,
            problem,
            gold,
            done_variants,
            experiment=experiment,
            with_semble_only=with_semble_only,
        )
        all_rows.append({"instance_id": iid, "gold_files": gold, "results": [asdict(r) for r in results]})

        if task_i < len(tasks) - 1:
            print(f"  (sleeping {_TASK_SLEEP}s...)\n")
            time.sleep(_TASK_SLEEP)

    _print_summary(all_rows, backend.label(), experiment=experiment)
    _save_outputs(all_rows, backend.label(), experiment=experiment)


def _print_summary(rows: list[dict], model_label: str, experiment: str | None = None) -> None:
    with_variant = f"with_semble_{experiment}" if experiment else "with_semble"
    with_r = [r for row in rows for r in row["results"] if r["variant"] == with_variant and not r["error"]]
    without_r_all = [r for row in rows for r in row["results"] if r["variant"] == "without_semble" and not r["error"]]
    without_r = [r for r in without_r_all if not r["bypass"]]
    n_bypass = len(without_r_all) - len(without_r)

    def avg(items: list[dict], key: str) -> float:
        vals = [r[key] for r in items]
        return sum(vals) / len(vals) if vals else 0.0

    def avg_semble_calls(items: list[dict]) -> float:
        vals = [sum(1 for t in r["tool_calls"] if is_semble_tool_call(t)) for r in items]
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
        cost_w_ci = bootstrap_ci([r["cost_usd"] for r in with_r])
        cost_wo_ci = bootstrap_ci([r["cost_usd"] for r in without_r])
        print(f"  {'avg cost (USD)':<25}  {avg(with_r, 'cost_usd'):>12.3f}  {avg(without_r, 'cost_usd'):>9.3f}")
        print(f"  {'  95% CI':<25}  [{cost_w_ci[0]:.3f},{cost_w_ci[1]:.3f}]  [{cost_wo_ci[0]:.3f},{cost_wo_ci[1]:.3f}]")
        print(
            f"  {'avg output tokens':<25}  "
            f"{avg(with_r, 'output_tokens'):>12.0f}  {avg(without_r, 'output_tokens'):>9.0f}"
        )
        print(f"  {'avg turns':<25}  {avg(with_r, 'num_turns'):>12.1f}  {avg(without_r, 'num_turns'):>9.1f}")
        print(f"  {'avg semble calls':<25}  {avg_semble_calls(with_r):>12.1f}  {avg_semble_calls(without_r):>9.1f}")
        ws_hit = sum(1 for r in with_r if r["gold_hit"]) / len(with_r)
        wo_hit = sum(1 for r in without_r if r["gold_hit"]) / len(without_r)
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
        ws_hit = sum(1 for r in with_r if r["gold_hit"]) / len(with_r)
        print(f"  {'gold file hit rate':<25}  {ws_hit:>12.0%}")
    errors_w = sum(1 for row in rows for r in row["results"] if r["variant"] == with_variant and r["error"])
    errors_wo = sum(1 for row in rows for r in row["results"] if r["variant"] == "without_semble" and r["error"])
    if errors_w or errors_wo:
        print(f"  {'errors':<25}  {errors_w:>12}  {errors_wo:>9}")


def _save_outputs(rows: list[dict], model_label: str, experiment: str | None = None) -> None:
    out = RESULTS_DIR / (f"swe_agent_{experiment}.json" if experiment else "swe_agent.json")
    existing: dict[str, dict] = {}
    if out.exists():
        for entry in json.loads(out.read_text()):
            existing[entry["instance_id"]] = entry
    for row in rows:
        iid = row["instance_id"]
        if iid in existing:
            result_map = {(r.get("backend", ""), r.get("model", ""), r["variant"]): r for r in existing[iid]["results"]}
            for new_r in row["results"]:
                key = (new_r.get("backend", ""), new_r.get("model", ""), new_r["variant"])
                result_map[key] = new_r
            existing[iid]["results"] = list(result_map.values())
        else:
            existing[iid] = row
    merged = list(existing.values())
    out.write_text(json.dumps(merged, indent=2))
    print(f"\nFull results -> {out}  ({len(merged)} instances)")

    suffix = f"_{experiment}" if experiment else ""
    model_slug = model_label.replace("/", "-")
    for variant in ("with_semble", "without_semble"):
        result_variant = f"{variant}_{experiment}" if experiment and variant == "with_semble" else variant
        predictions = [
            {
                "instance_id": entry["instance_id"],
                "model_patch": r["patch"],
                "model_name_or_path": f"{model_slug}-{variant}",
            }
            for entry in merged
            for r in entry["results"]
            if r["variant"] == result_variant and not r["error"] and r["patch"] and not r.get("bypass")
        ]
        path = RESULTS_DIR / f"predictions_{variant}{suffix}.jsonl"
        path.write_text("\n".join(json.dumps(p) for p in predictions) + "\n")
        print(f"Predictions ({variant}): {path}  ({len(predictions)} patches)")

    ids = sorted(entry["instance_id"] for entry in merged)
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
        default=_DEFAULT_REPO,
        help="Repo name, comma-separated repo names, or 'all' to sample across the whole dataset",
    )
    p.add_argument("--seed", type=int, default=_DEFAULT_SEED, help="Seed for random task sampling")
    p.add_argument(
        "--refresh-cache", action="store_true", help="Re-fetch the SWE-bench Lite task list from HuggingFace"
    )
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
        refresh_cache=args.refresh_cache,
    )


if __name__ == "__main__":
    main()
