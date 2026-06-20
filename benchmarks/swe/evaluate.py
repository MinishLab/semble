from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from benchmarks.swe.stats import mcnemar_exact_p

RESULTS_DIR = Path(__file__).parent / "results"
_PROJECT_ROOT = Path(__file__).parent.parent.parent
_DATASET = "princeton-nlp/SWE-bench_Lite"


def _check_docker() -> None:
    """Exit if Docker is not running."""
    try:
        subprocess.run(["docker", "info"], capture_output=True, check=True, timeout=10)
    except (subprocess.CalledProcessError, FileNotFoundError):
        sys.exit("Docker is not running. Start OrbStack and retry.")


def _run_harness(predictions_path: Path, instance_ids: list[str], run_id: str) -> dict[str, bool]:
    """Run the harness and return ``{instance_id: resolved}`` map."""
    lines = [ln for ln in predictions_path.read_text().splitlines() if ln.strip()]
    if not lines:
        print(f"  Skipping {run_id} — no predictions in {predictions_path.name}")
        return {}

    pred_ids = {json.loads(ln)["instance_id"] for ln in lines}
    ids_to_run = [i for i in instance_ids if i in pred_ids]
    if not ids_to_run:
        print(f"  Skipping {run_id} — none of the requested instance IDs have patches")
        return {}

    cmd = [
        "uv",
        "run",
        "--with",
        "swebench",
        "python",
        "-m",
        "swebench.harness.run_evaluation",
        "--dataset_name",
        _DATASET,
        "--predictions_path",
        str(predictions_path),
        "--max_workers",
        "1",
        "--run_id",
        run_id,
        "--instance_ids",
    ] + ids_to_run

    print(f"\nRunning harness: {run_id}  ({len(ids_to_run)} instances)")
    subprocess.run(cmd, check=True, cwd=_PROJECT_ROOT)

    model_name = json.loads(lines[0])["model_name_or_path"]
    result_file = _PROJECT_ROOT / f"{model_name}.{run_id}.json"
    if not result_file.exists():
        candidates = list(_PROJECT_ROOT.glob(f"*.{run_id}.json"))
        if not candidates:
            print(f"  Warning: result file not found for {run_id}")
            return {}
        result_file = candidates[0]

    data = json.loads(result_file.read_text())
    result_file.rename(RESULTS_DIR / result_file.name)

    return {
        **{iid: True for iid in data.get("resolved_ids", [])},
        **{iid: False for iid in data.get("unresolved_ids", [])},
        **{iid: False for iid in data.get("empty_patch_ids", [])},
    }


def _paired_summary(results: dict[str, dict[str, bool]], instance_ids: list[str]) -> None:
    """McNemar's test on paired with/without resolve outcomes."""
    paired = [iid for iid in instance_ids if iid in results["with_semble"] and iid in results["without_semble"]]
    if not paired:
        return
    b = sum(1 for iid in paired if results["with_semble"][iid] and not results["without_semble"][iid])
    c = sum(1 for iid in paired if results["without_semble"][iid] and not results["with_semble"][iid])
    both = sum(1 for iid in paired if results["with_semble"][iid] and results["without_semble"][iid])
    neither = len(paired) - b - c - both
    p = mcnemar_exact_p(b, c)
    print(f"\n  Paired comparison (n={len(paired)} tasks with both variants evaluated):")
    print(f"    resolved by both:        {both}")
    print(f"    resolved by neither:     {neither}")
    print(f"    resolved with only:      {b}")
    print(f"    resolved without only:   {c}")
    print(f"    McNemar exact p-value:   {p:.4f}")
    if len(paired) < 30:
        print(f"    (n={len(paired)} is too small to draw a conclusion from this p-value either way)")


def _collect_instance_ids(*pred_paths: Path) -> list[str]:
    """Union of all instance IDs found across the given prediction files."""
    ids: set[str] = set()
    for p in pred_paths:
        for line in p.read_text().splitlines():
            if line.strip():
                ids.add(json.loads(line)["instance_id"])
    return sorted(ids)


def _model_slug(pred_path: Path, fallback: str) -> str:
    """Model name to scope the harness run_id by, so different backend runs don't share harness logs."""
    first_line = next((ln for ln in pred_path.read_text().splitlines() if ln.strip()), None)
    return json.loads(first_line)["model_name_or_path"] if first_line else fallback


def _fmt_resolved(v: bool | None) -> str:
    return "✓" if v is True else ("✗" if v is False else "?")


def _print_resolve_table(results: dict[str, dict[str, bool]], all_ids: list[str]) -> None:
    print(f"\n{'=' * 58}")
    print("Resolve rate — real SWE-bench test verification")
    print(f"{'=' * 58}")
    print(f"  {'Instance':<40}  {'With':>5}  {'Without':>8}")
    print(f"  {'-' * 40}  {'-' * 5}  {'-' * 8}")
    for iid in all_ids:
        w = results["with_semble"].get(iid)
        wo = results["without_semble"].get(iid)
        print(f"  {iid:<40}  {_fmt_resolved(w):>5}  {_fmt_resolved(wo):>8}")

    print()
    for variant, label in [("with_semble", "With Semble"), ("without_semble", "Without Semble")]:
        r = results.get(variant, {})
        if r:
            resolved = sum(r.values())
            print(f"  {label}: {resolved}/{len(r)} resolved ({resolved / len(r):.0%})")


def run(instance_ids: list[str] | None = None, experiment: str | None = None) -> None:
    """Evaluate with/without-semble predictions through the real SWE-bench Docker harness."""
    _check_docker()
    RESULTS_DIR.mkdir(exist_ok=True)

    suffix = f"_{experiment}" if experiment else ""
    pred_with = RESULTS_DIR / f"predictions_with_semble{suffix}.jsonl"
    pred_without = RESULTS_DIR / "predictions_without_semble.jsonl"
    for p in (pred_with, pred_without):
        if not p.exists():
            sys.exit(f"Missing {p} — run agent_run.py first")

    if not instance_ids:
        instance_ids = _collect_instance_ids(pred_with, pred_without)

    print(f"Evaluating {len(instance_ids)} instances with real SWE-bench harness...")

    results: dict[str, dict[str, bool]] = {}
    for pred_path, variant in [(pred_with, "with_semble"), (pred_without, "without_semble")]:
        run_id = f"semble_eval_{_model_slug(pred_path, variant)}"
        results[variant] = _run_harness(pred_path, instance_ids, run_id)

    all_ids = sorted(set().union(*[set(r.keys()) for r in results.values()]))
    _print_resolve_table(results, all_ids)
    _paired_summary(results, all_ids)

    out = RESULTS_DIR / f"swe_resolve{suffix}.json"
    out.write_text(json.dumps({"instances": all_ids, "results": results}, indent=2))
    print(f"\nSaved -> {out}")


def main() -> None:
    """Parse CLI arguments and run the SWE-bench evaluation harness."""
    p = argparse.ArgumentParser()
    p.add_argument("--instance-ids", nargs="*", help="Specific IDs to evaluate (default: all in prediction files)")
    p.add_argument(
        "--experiment",
        default=None,
        help="Read predictions_with_semble_{NAME}.jsonl instead of the default with-semble predictions",
    )
    args = p.parse_args()
    run(args.instance_ids, args.experiment)


if __name__ == "__main__":
    main()
