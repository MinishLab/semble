from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from swebench.harness.constants import KEY_INSTANCE_ID, KEY_MODEL
from swebench.harness.utils import get_predictions_from_file

from benchmarks.swe.utils import (
    DATASET,
    PROJECT_ROOT,
    RESULTS_DIR,
    WITH_SEMBLE,
    WITHOUT_SEMBLE,
    mcnemar_exact_p,
    prediction_path,
    resolve_results_path,
)


def _check_docker() -> None:
    """Exit if Docker is not running."""
    try:
        subprocess.run(["docker", "info"], capture_output=True, check=True, timeout=10)
    except (subprocess.CalledProcessError, FileNotFoundError):
        sys.exit("Docker is not running. Start OrbStack and retry.")


def _run_harness(predictions_path: Path, instance_ids: list[str], run_id: str) -> dict[str, bool]:
    """Run the harness and return ``{instance_id: resolved}`` map."""
    predictions = get_predictions_from_file(str(predictions_path), DATASET, "test")
    if not predictions:
        print(f"  Skipping {run_id} — no predictions in {predictions_path.name}")
        return {}

    pred_ids = {p[KEY_INSTANCE_ID] for p in predictions}
    ids_to_run = [i for i in instance_ids if i in pred_ids]
    if not ids_to_run:
        print(f"  Skipping {run_id} — none of the requested instance IDs have patches")
        return {}

    cmd = [
        "uv",
        "run",
        "python",
        "-m",
        "swebench.harness.run_evaluation",
        "--dataset_name",
        DATASET,
        "--predictions_path",
        str(predictions_path),
        "--max_workers",
        "1",
        "--run_id",
        run_id,
        "--instance_ids",
    ] + ids_to_run

    print(f"\nRunning harness: {run_id}  ({len(ids_to_run)} instances)")
    subprocess.run(cmd, check=True, cwd=PROJECT_ROOT)

    model_name = predictions[0][KEY_MODEL]
    result_file = PROJECT_ROOT / f"{model_name}.{run_id}.json"
    if not result_file.exists():
        candidates = list(PROJECT_ROOT.glob(f"*.{run_id}.json"))
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
    paired = [iid for iid in instance_ids if iid in results[WITH_SEMBLE] and iid in results[WITHOUT_SEMBLE]]
    if not paired:
        return
    b = sum(1 for iid in paired if results[WITH_SEMBLE][iid] and not results[WITHOUT_SEMBLE][iid])
    c = sum(1 for iid in paired if results[WITHOUT_SEMBLE][iid] and not results[WITH_SEMBLE][iid])
    both = sum(1 for iid in paired if results[WITH_SEMBLE][iid] and results[WITHOUT_SEMBLE][iid])
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
        for pred in get_predictions_from_file(str(p), DATASET, "test"):
            ids.add(pred[KEY_INSTANCE_ID])
    return sorted(ids)


def _model_slug(pred_path: Path, fallback: str) -> str:
    """Model name to scope the harness run_id by, so different backend runs don't share harness logs."""
    preds = get_predictions_from_file(str(pred_path), DATASET, "test")
    return preds[0][KEY_MODEL] if preds else fallback


def _fmt_resolved(v: bool | None) -> str:
    return "✓" if v is True else ("✗" if v is False else "?")


def _print_resolve_table(results: dict[str, dict[str, bool]], all_ids: list[str]) -> None:
    print(f"\n{'=' * 58}")
    print("Resolve rate — real SWE-bench test verification")
    print(f"{'=' * 58}")
    print(f"  {'Instance':<40}  {'With':>5}  {'Without':>8}")
    print(f"  {'-' * 40}  {'-' * 5}  {'-' * 8}")
    for iid in all_ids:
        w = results[WITH_SEMBLE].get(iid)
        wo = results[WITHOUT_SEMBLE].get(iid)
        print(f"  {iid:<40}  {_fmt_resolved(w):>5}  {_fmt_resolved(wo):>8}")

    print()
    for variant, label in [(WITH_SEMBLE, "With Semble"), (WITHOUT_SEMBLE, "Without Semble")]:
        r = results.get(variant, {})
        if r:
            resolved = sum(r.values())
            print(f"  {label}: {resolved}/{len(r)} resolved ({resolved / len(r):.0%})")


def run(instance_ids: list[str] | None = None, experiment: str | None = None, model: str | None = None) -> None:
    """Evaluate with/without-semble predictions through the real SWE-bench Docker harness."""
    _check_docker()
    RESULTS_DIR.mkdir(exist_ok=True)

    model_slug = model.replace("/", "-") if model else None
    pred_with = prediction_path(with_semble=True, experiment=experiment, model_slug=model_slug)
    pred_without = prediction_path(with_semble=False, model_slug=model_slug)
    for p in (pred_with, pred_without):
        if not p.exists():
            sys.exit(f"Missing {p} — run agent_run.py first")

    if not instance_ids:
        instance_ids = _collect_instance_ids(pred_with, pred_without)

    print(f"Evaluating {len(instance_ids)} instances with real SWE-bench harness...")

    results: dict[str, dict[str, bool]] = {}
    for pred_path, variant in [(pred_with, WITH_SEMBLE), (pred_without, WITHOUT_SEMBLE)]:
        run_id = f"semble_eval_{_model_slug(pred_path, variant)}"
        results[variant] = _run_harness(pred_path, instance_ids, run_id)

    all_ids = sorted(set().union(*[set(r.keys()) for r in results.values()]))
    _print_resolve_table(results, all_ids)
    _paired_summary(results, all_ids)

    out = resolve_results_path(experiment)
    out.write_text(json.dumps({"instances": all_ids, "results": results}, indent=2))
    print(f"\nSaved -> {out}")


def main() -> None:
    """Parse CLI arguments and run the SWE-bench evaluation harness."""
    p = argparse.ArgumentParser()
    p.add_argument("--instance-ids", nargs="*", help="Specific IDs to evaluate (default: all in prediction files)")
    p.add_argument(
        "--experiment",
        default=None,
        help="Read the with-semble predictions for this --experiment instead of the default run",
    )
    p.add_argument(
        "--model",
        default=None,
        help="Backend/model label (e.g. codex/gpt-5.4-mini) used by agent_run.py for this run. "
        "Required to find the matching predictions_{with,without}_semble_{model}*.jsonl files "
        "(omit only for legacy unscoped files from before this scoping existed).",
    )
    args = p.parse_args()
    run(args.instance_ids, args.experiment, args.model)


if __name__ == "__main__":
    main()
