import argparse
import json
import re
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, TypeAlias

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import tiktoken
from model2vec import StaticModel

from benchmarks.data import (
    Target,
    Task,
    add_filter_args,
    grouped_tasks,
    load_filtered_tasks,
    save_results,
    target_matches_location,
)
from semble import SembleIndex
from semble.index.dense import _DEFAULT_MODEL_NAME
from semble.index.file_walker import DEFAULT_IGNORED_DIRS, FILE_TYPES, FileCategory
from semble.ranking.boosting import _STOPWORDS as _SEMBLE_STOPWORDS
from semble.types import Chunk

_RG_INCLUDE_GLOBS: tuple[str, ...] = tuple(
    f"*{ext}" for ext, spec in FILE_TYPES.items() if spec.category == FileCategory.CODE
)
_RG_EXCLUDE_GLOBS: tuple[str, ...] = tuple(f"!{d}" for d in DEFAULT_IGNORED_DIRS)

_BUDGETS = (500, 1000, 2000, 4000, 8000, 16000, 32000)
_EXPECTED_COST_CAP = 32_000
_PLOT_BUDGETS = sorted({int(b) for b in np.logspace(np.log10(100), np.log10(256000), 60)})
_TOKENIZER_NAME = "cl100k_base"
_RG_MAX_MATCHES = 500
_SEMBLE_TOP_K = 50
_KW_MIN_LEN = 3

# Extend semble's code-ranking stopwords with broader NL query words.
_STOPWORDS: frozenset[str] = _SEMBLE_STOPWORDS | frozenset(
    """
    but its can not no nor so yet both either neither than then
    will would could should may might been being had did will
    they all any each few more most other some such only own same too very just
    about after also before between during into through under up down over
    """.split()
)

_IMAGES_DIR = Path(__file__).parent.parent / "assets" / "images"
_RESULTS_DIR = Path(__file__).parent / "results"

Curve: TypeAlias = list[tuple[int, int]]
MethodCurves: TypeAlias = list[tuple[Curve, int]]


def _semble_units(index: SembleIndex, query: str) -> list[Chunk]:
    """Top-K semble chunks as units, ordered by score."""
    return [r.chunk for r in index.search(query, top_k=_SEMBLE_TOP_K)]


def _rg_command(pattern: str, repo_dir: Path) -> list[str]:
    """Build the rg command line, scoped to the same code-file universe semble indexes."""
    cmd = ["rg", "--json", "--fixed-strings", "--ignore-case"]
    for glob in (*_RG_EXCLUDE_GLOBS, *_RG_INCLUDE_GLOBS):
        cmd += ["--glob", glob]
    cmd += [pattern, str(repo_dir)]
    return cmd


def _rg_matches(pattern: str, repo_dir: Path) -> list[tuple[str, int]]:
    """Return (file_path, line_number) matches via rg --json, in rg's output order."""
    try:
        proc = subprocess.run(
            _rg_command(pattern, repo_dir),
            capture_output=True,
            text=True,
            timeout=30,
        )
    except subprocess.TimeoutExpired:
        return []
    if proc.returncode not in (0, 1):
        return []
    matches: list[tuple[str, int]] = []
    for line in proc.stdout.splitlines():
        if not line:
            continue
        try:
            evt = json.loads(line)
        except json.JSONDecodeError:
            continue
        if evt.get("type") != "match":
            continue
        data = evt.get("data", {})
        path = data.get("path", {}).get("text")
        ln = data.get("line_number")
        if path and isinstance(ln, int):
            matches.append((path, ln))
    return matches


def _grep_file_units(
    pattern: str,
    repo_dir: Path,
) -> list[Chunk]:
    """Return whole matched files in match-count order."""
    matches = _rg_matches(pattern, repo_dir)
    if not matches:
        return []
    ranked = sorted(Counter(path for path, _ in matches[:_RG_MAX_MATCHES]).items(), key=lambda kv: (-kv[1], kv[0]))
    units: list[Chunk] = []
    for path, _ in ranked:
        try:
            text = Path(path).read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        units.append(Chunk(content=text, file_path=path, start_line=1, end_line=text.count("\n") + 1))
    return units


def _keywords(query: str) -> list[str]:
    """Extract meaningful search keywords from a natural-language query."""
    words = re.findall(r"[a-zA-Z][a-zA-Z0-9]*", query)
    seen: set[str] = set()
    result: list[str] = []
    for w in words:
        lw = w.lower()
        if len(lw) >= _KW_MIN_LEN and lw not in _STOPWORDS and lw not in seen:
            seen.add(lw)
            result.append(w)
    return result


def _grep_keywords_file_units(query: str, repo_dir: Path) -> list[Chunk]:
    """Return files ranked by how many distinct query keywords they contain."""
    keywords = _keywords(query)
    if not keywords:
        return _grep_file_units(query, repo_dir)
    keyword_hits: Counter[str] = Counter()
    for kw in keywords:
        for path, _ in _rg_matches(kw, repo_dir)[:_RG_MAX_MATCHES]:
            keyword_hits[path] += 1
    ranked = sorted(keyword_hits.items(), key=lambda kv: (-kv[1], kv[0]))
    units: list[Chunk] = []
    for path, _ in ranked:
        try:
            text = Path(path).read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        units.append(Chunk(content=text, file_path=path, start_line=1, end_line=text.count("\n") + 1))
    return units


def _retrieval_units_for_task(
    index: SembleIndex,
    task: Task,
    repo_dir: Path,
) -> list[tuple[str, list[Chunk]]]:
    """Return (method, units) pairs for a task."""
    return [
        ("semble", _semble_units(index, task.query)),
        ("grep+read", _grep_file_units(task.query, repo_dir)),
        ("grep-kw+read", _grep_keywords_file_units(task.query, repo_dir)),
    ]


def _curve(units: list[Chunk], targets: tuple[Target, ...], enc: Any) -> Curve:
    """Cumulative (tokens, covered_target_count) after each retrieved unit, starting at (0, 0)."""
    covered = [False] * len(targets)
    cumulative = 0
    points: Curve = [(0, 0)]
    for unit in units:
        cumulative += len(enc.encode(unit.content, disallowed_special=()))
        for i, tgt in enumerate(targets):
            if not covered[i] and target_matches_location(unit.file_path, unit.start_line, unit.end_line, tgt):
                covered[i] = True
        points.append((cumulative, sum(covered)))
    return points


def _recall_at(curve: Curve, budget: int, n_total: int) -> float:
    """Recall (covered_targets / n_total) at the largest cumulative-tokens point <= budget."""
    if n_total == 0 or not curve:
        return 0.0
    covered_at = 0
    for tokens, covered in curve:
        if tokens > budget:
            break
        covered_at = covered
    return covered_at / n_total


def _mean_recall_at(curves: MethodCurves, budgets: tuple[int, ...] | list[int]) -> dict[int, float]:
    """Mean recall at each budget across all queries."""
    return {b: float(np.mean([_recall_at(c, b, n) for c, n in curves if n > 0])) if curves else 0.0 for b in budgets}


def _mean_curve(curves: MethodCurves, grid: list[int]) -> list[float]:
    """Mean recall over all queries at each grid budget."""
    samples = [[_recall_at(c, b, n) for b in grid] for c, n in curves if n > 0]
    return np.mean(samples, axis=0).tolist() if samples else [0.0] * len(grid)


def _tokens_to_first_hit(curve: Curve) -> int | None:
    """Cumulative tokens at which the first relevant target is covered, or None if never."""
    for tokens, covered in curve:
        if covered > 0:
            return tokens
    return None


def _expected_cost_at_cap(curves: MethodCurves, cap: int) -> float:
    """Mean tokens spent before first hit or giving up at cap, across all queries."""
    costs = [hit if (hit := _tokens_to_first_hit(c)) is not None else cap for c, n in curves if n > 0]
    return float(np.mean(costs)) if costs else float(cap)


def _pairwise_reduction(semble: MethodCurves, other: MethodCurves) -> dict[str, float] | None:
    """Median 'tokens to first hit' reduction, paired on queries where both methods hit."""
    pairs: list[tuple[int, int]] = []
    for (s_curve, _), (o_curve, _) in zip(semble, other, strict=True):
        s = _tokens_to_first_hit(s_curve)
        o = _tokens_to_first_hit(o_curve)
        if s is not None and o is not None and o > 0:
            pairs.append((s, o))
    if not pairs:
        return None
    ratios = [s / o for s, o in pairs]
    return {
        "n_paired": float(len(pairs)),
        "median_semble_tokens": float(np.median([s for s, _ in pairs])),
        "median_other_tokens": float(np.median([o for _, o in pairs])),
        "median_reduction": 1.0 - float(np.median(ratios)),
        "mean_reduction": 1.0 - float(np.mean(ratios)),
        "semble_better_pct": sum(1 for s, o in pairs if s < o) / len(pairs),
    }


def _evaluate_repo_recall(
    index: SembleIndex,
    tasks: list[Task],
    repo_dir: Path,
    enc: Any,
) -> dict[str, MethodCurves]:
    """Build per-method curves for every task in the repo."""
    methods: dict[str, MethodCurves] = defaultdict(list)
    for task in tasks:
        targets = task.all_relevant
        n = len(targets)
        for method, units in _retrieval_units_for_task(index, task, repo_dir):
            methods[method].append((_curve(units, targets, enc), n))
    return dict(methods)


_PLOT_STYLE: dict[str, dict[str, object]] = {
    "semble": {"label": "semble", "color": "#1a5fa8", "linewidth": 2.4, "zorder": 4},
    "grep-kw+read": {"label": "ripgrep + read", "color": "#b7770d", "linewidth": 1.8, "zorder": 3},
}


_PLOT_MAX_BUDGET = 100_000


def _plot_recall_vs_tokens(payload: dict[str, Any], out_path: Path) -> None:
    """Render a recall-vs-tokens curve from a recall-mode payload."""
    plot_data = payload["plot"]
    all_budgets = plot_data["budgets"]
    all_recalls = plot_data["recall"]

    budgets = all_budgets
    recalls = all_recalls

    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.grid(axis="y", color="#e8e8e8", linewidth=0.7, zorder=0)
    ax.grid(axis="x", color="#f0f0f0", linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color("#cccccc")

    for method, style in _PLOT_STYLE.items():
        if method not in recalls:
            continue
        ax.plot(
            budgets,
            recalls[method],
            label=style["label"],
            color=style["color"],
            linewidth=style["linewidth"],
            linestyle=style.get("linestyle", "-"),
            zorder=style["zorder"],
        )

    ax.set_xlim(0, _PLOT_MAX_BUDGET)
    ax.set_ylim(0.0, 1.02)
    ax.set_xlabel("Retrieved context tokens", fontsize=10, color="#444444")
    ax.set_ylabel("Recall (relevant files surfaced)", fontsize=10, color="#444444")
    ax.set_title(
        "Token efficiency: recall vs. retrieved tokens",
        fontsize=12,
        color="#222222",
        pad=12,
    )
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v / 1000:.0f}k" if v >= 1000 else f"{v:.0f}"))
    ax.tick_params(labelsize=9, colors="#555555")
    ax.legend(loc="lower right", fontsize=9, frameon=True, framealpha=0.95, edgecolor="#dddddd")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved plot to {out_path}", file=sys.stderr)


def _print_recall_summary(method_curves: dict[str, MethodCurves]) -> dict[str, dict[int, float]]:
    """Print and return recall@budget per method."""
    print(f"\nTokenizer: {_TOKENIZER_NAME}\n", file=sys.stderr)
    print(f"{'Method':<20} " + "  ".join(f"{b:>7}" for b in _BUDGETS), file=sys.stderr)
    print(f"{'-' * 20} " + "  ".join(f"{'-' * 7:>7}" for _ in _BUDGETS), file=sys.stderr)
    summary: dict[str, dict[int, float]] = {}
    for method, curves in method_curves.items():
        recall = _mean_recall_at(curves, _BUDGETS)
        summary[method] = recall
        print(f"{method:<20} " + "  ".join(f"{recall[b]:>7.3f}" for b in _BUDGETS), file=sys.stderr)
    return summary


def _print_first_hit_summary(method_curves: dict[str, MethodCurves]) -> dict[str, dict[str, float]]:
    """Print and return pairwise tokens-to-first-hit reductions vs semble."""
    print("\nTokens to first relevant file (semble vs other, paired)", file=sys.stderr)
    print(f"{'vs':<20} {'n':>5}  {'med-semble':>10}  {'med-other':>10}  {'med-reduce':>10}", file=sys.stderr)
    print(f"{'-' * 20} {'-' * 5}  {'-' * 10}  {'-' * 10}  {'-' * 10}", file=sys.stderr)
    reductions: dict[str, dict[str, float]] = {}
    for method, curves in method_curves.items():
        if method == "semble":
            continue
        red = _pairwise_reduction(method_curves["semble"], curves)
        if red is None:
            continue
        reductions[method] = red
        print(
            f"{method:<20} {int(red['n_paired']):>5}  "
            f"{red['median_semble_tokens']:>10.0f}  {red['median_other_tokens']:>10.0f}  "
            f"{red['median_reduction']:>10.1%}",
            file=sys.stderr,
        )

    print(
        f"\nExpected tokens per query (first hit or {_EXPECTED_COST_CAP // 1000}k cap if no hit)",
        file=sys.stderr,
    )
    print(f"{'Method':<20} {'expected-tokens':>15}  {'vs-semble':>10}", file=sys.stderr)
    print(f"{'-' * 20} {'-' * 15}  {'-' * 10}", file=sys.stderr)
    semble_cost = _expected_cost_at_cap(method_curves["semble"], _EXPECTED_COST_CAP)
    print(f"{'semble':<20} {semble_cost:>15.0f}", file=sys.stderr)
    for method, curves in method_curves.items():
        if method == "semble":
            continue
        cost = _expected_cost_at_cap(curves, _EXPECTED_COST_CAP)
        ratio = cost / semble_cost if semble_cost > 0 else float("inf")
        print(f"{method:<20} {cost:>15.0f}  {ratio:>9.1f}x", file=sys.stderr)
        reductions[method]["expected_cost_at_cap"] = cost
        reductions[method]["expected_cost_cap"] = float(_EXPECTED_COST_CAP)
        reductions[method]["expected_cost_ratio_vs_semble"] = ratio
    reductions["semble"] = {"expected_cost_at_cap": semble_cost, "expected_cost_cap": float(_EXPECTED_COST_CAP)}

    return reductions


def run_recall(args: argparse.Namespace) -> None:
    """Run the recall-vs-token-budget benchmark."""
    repo_specs, tasks = load_filtered_tasks(args.repo or None, args.language or None)

    print("Loading tokenizer + model...", file=sys.stderr)
    enc = tiktoken.get_encoding(_TOKENIZER_NAME)
    model = StaticModel.from_pretrained(_DEFAULT_MODEL_NAME)

    method_curves: dict[str, MethodCurves] = defaultdict(list)
    print(f"\n{'Repo':<22} {'Language':<12} {'Tasks':>6} {'Time':>8}", file=sys.stderr)
    print(f"{'-' * 22} {'-' * 12} {'-' * 6} {'-' * 8}", file=sys.stderr)
    for repo, repo_task_list in sorted(grouped_tasks(tasks).items()):
        spec = repo_specs[repo]
        started = time.perf_counter()
        index = SembleIndex.from_path(spec.benchmark_dir, model=model)
        per_method = _evaluate_repo_recall(index, repo_task_list, spec.benchmark_dir, enc)
        for m, lst in per_method.items():
            method_curves[m].extend(lst)
        print(
            f"{repo:<22} {spec.language:<12} {len(repo_task_list):>6} {time.perf_counter() - started:>7.1f}s",
            file=sys.stderr,
        )

    summary = _print_recall_summary(method_curves)
    reductions = _print_first_hit_summary(method_curves)

    payload: dict[str, Any] = {
        "tool": "token-efficiency",
        "tokenizer": _TOKENIZER_NAME,
        "budgets": list(_BUDGETS),
        "n_queries": len(method_curves["semble"]),
        "recall_at_budget": {m: {str(b): round(v, 4) for b, v in r.items()} for m, r in summary.items()},
        "first_hit_reduction": {m: {k: round(v, 4) for k, v in d.items()} for m, d in reductions.items()},
        "plot": {
            "budgets": _PLOT_BUDGETS,
            "recall": {m: [round(x, 4) for x in _mean_curve(c, _PLOT_BUDGETS)] for m, c in method_curves.items()},
        },
    }
    if args.repo or args.language:
        print(json.dumps(payload, indent=2))
        return

    out = save_results("token-efficiency", payload)
    print(f"\nResults saved to {out}", file=sys.stderr)
    if not args.no_plot:
        _IMAGES_DIR.mkdir(parents=True, exist_ok=True)
        _plot_recall_vs_tokens(payload, _IMAGES_DIR / "token_efficiency.png")


def run_plot(args: argparse.Namespace) -> None:
    """Plot recall-vs-tokens from a saved recall-mode payload."""
    matches = sorted(_RESULTS_DIR.glob("token-efficiency-*.json"))
    in_path = args.input or (matches[-1] if matches else None)
    if in_path is None:
        raise SystemExit(f"No recall results found in {_RESULTS_DIR}")
    payload = json.loads(in_path.read_text(encoding="utf-8"))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    _plot_recall_vs_tokens(payload, args.output)


def _parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Context-efficiency benchmark: semble vs grep workflows.")
    parser.set_defaults(func=run_recall, repo=[], language=[], no_plot=False)
    sub = parser.add_subparsers(dest="mode", required=False)

    recall = sub.add_parser("recall", help="Recall vs. token-budget across all queries (default).")
    add_filter_args(recall)
    recall.add_argument("--no-plot", action="store_true", help="Skip plotting after the run.")
    recall.set_defaults(func=run_recall)

    plot = sub.add_parser("plot", help="Regenerate the recall-vs-tokens plot from a saved JSON.")
    plot.add_argument("--input", type=Path, default=None, help="Path to recall results (default: newest).")
    plot.add_argument("--output", type=Path, default=_IMAGES_DIR / "token_efficiency.png", help="Output PNG path.")
    plot.set_defaults(func=run_plot)

    return parser.parse_args()


def main() -> None:
    """Dispatch to the requested mode."""
    args = _parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
