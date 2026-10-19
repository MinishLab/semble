"""Train the learned ranker (src/semble/ranking/ranker.json) on the benchmark annotations.

LightGBM is only needed here, not at query time:

    uv run --with lightgbm python -m benchmarks.train_ranker --cv   # leave-one-repo-out NDCG@10, no file written
    uv run --with lightgbm python -m benchmarks.train_ranker        # train on every repo and write ranker.json

For each query, the candidate pool and features come from the same code path search uses
(``score_candidates`` + ``ranking_features``). The model is a LambdaRank tree ensemble that learns a log-scale
correction to the boosted score (``init_score = log(boosted / max boosted)``), so an empty model reproduces
today's ranking. Only the best-ranked chunk covering each annotated target is labelled relevant: labelling every
chunk of a whole-file target teaches the model to promote large files.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

from benchmarks.data import add_filter_args, grouped_tasks, load_filtered_tasks, target_matches_location
from benchmarks.metrics import ndcg_at_k, target_rank
from semble import SembleIndex
from semble.ranking import rerank_topk
from semble.ranking.learned import FEATURES, RankingSignals, TreeEnsemble, ranking_features
from semble.search import score_candidates
from semble.types import SearchResult

_TOP_K = 10
_OUT = Path(__file__).resolve().parent.parent / "src" / "semble" / "ranking" / "ranker.json"
_PARAMS = {
    "objective": "lambdarank",
    "n_estimators": 150,
    "learning_rate": 0.03,
    "num_leaves": 7,
    "min_child_samples": 200,
    "reg_lambda": 10.0,
    "lambdarank_truncation_level": 10,
    "deterministic": True,
    "force_row_wise": True,
    "verbose": -1,
}


def _collect(repo_filter: list[str] | None, language_filter: list[str] | None) -> list[dict]:
    """Return one record per query: repo, task, signals, pool, features and first-hit labels."""
    specs, tasks = load_filtered_tasks(repo_filter, language_filter)
    records = []
    for repo, repo_tasks in sorted(grouped_tasks(tasks).items()):
        index = SembleIndex.from_path(specs[repo].benchmark_dir)
        for task in repo_tasks:
            signals = score_candidates(
                task.query, index.model, index._semantic_index, index._bm25_index, index.chunks, _TOP_K
            )
            pool, features = ranking_features(signals)
            labels = np.zeros(len(pool), dtype=int)
            for target in task.all_relevant:
                for i, chunk in enumerate(pool):
                    if target_matches_location(chunk.file_path, chunk.start_line, chunk.end_line, target):
                        labels[i] = 1
                        break
            records.append({"repo": repo, "task": task, "signals": signals, "pool": pool, "x": features, "y": labels})
        print(f"  {repo}: {len(repo_tasks)} queries", file=sys.stderr)
    return records


def _init_score(x: np.ndarray) -> np.ndarray:
    return np.log(np.maximum(x[:, FEATURES.index("final_rel")], 1e-9))


def _fit(records: list[dict]):  # noqa: ANN202
    import lightgbm as lgb

    model = lgb.LGBMRanker(**_PARAMS)
    x = np.vstack([r["x"] for r in records])
    model.fit(
        x, np.concatenate([r["y"] for r in records]), group=[len(r["pool"]) for r in records], init_score=_init_score(x)
    )
    return model


def _ndcg(record: dict, scores: dict) -> float:
    signals: RankingSignals = record["signals"]
    ranked = rerank_topk(scores, _TOP_K, penalise_paths=signals.alpha < 1.0)
    results = [SearchResult(chunk=c, score=s) for c, s in ranked]
    task = record["task"]
    ranks = [r for t in task.all_relevant if (r := target_rank(results, t)) is not None]
    return ndcg_at_k(ranks, len(task.all_relevant), _TOP_K)


def _corrected(record: dict, correction: np.ndarray) -> dict:
    boosted = record["signals"].boosted
    return {c: boosted[c] * float(np.exp(d)) for c, d in zip(record["pool"], correction)}


def to_ensemble_dict(booster) -> dict:  # noqa: ANN001
    """Convert a LightGBM booster to ranker.json's flat per-tree node arrays."""
    trees = []
    for info in booster.dump_model()["tree_info"]:
        feature: list[int] = []
        threshold: list[float] = []
        left: list[int] = []
        right: list[int] = []
        value: list[float] = []

        def add(node: dict) -> int:
            nid = len(feature)
            feature.append(-1)
            threshold.append(0.0)
            left.append(0)
            right.append(0)
            value.append(0.0)  # noqa: E702
            if "leaf_value" in node:
                value[nid] = float(node["leaf_value"])
                return nid
            if node.get("decision_type", "<=") != "<=":
                raise ValueError(f"Unsupported split {node.get('decision_type')!r}")
            feature[nid] = int(node["split_feature"])
            threshold[nid] = float(node["threshold"])
            left[nid] = add(node["left_child"])
            right[nid] = add(node["right_child"])
            return nid

        add(info["tree_structure"])
        trees.append({"feature": feature, "threshold": threshold, "left": left, "right": right, "value": value})
    return {"features": list(FEATURES), "params": {k: v for k, v in _PARAMS.items() if k != "verbose"}, "trees": trees}


def main() -> None:
    """Train, cross-validate, or export the learned ranker."""
    parser = argparse.ArgumentParser(description="Train semble's learned ranker on the benchmark annotations.")
    add_filter_args(parser)
    parser.add_argument("--cv", action="store_true", help="Report leave-one-repo-out NDCG@10 instead of writing.")
    parser.add_argument("--output", type=Path, default=_OUT)
    args = parser.parse_args()

    # Features must come from the hand-tuned ranking only, never from a previously shipped model.
    os.environ["SEMBLE_LEARNED_RANKER"] = "0"
    records = _collect(args.repo or None, args.language or None)
    print(f"{len(records)} queries, {sum(len(r['pool']) for r in records)} candidates", file=sys.stderr)

    if args.cv:
        repos = sorted({r["repo"] for r in records})
        base, learned = [], []
        for repo in repos:
            model = _fit([r for r in records if r["repo"] != repo])
            for r in (r for r in records if r["repo"] == repo):
                base.append(_ndcg(r, r["signals"].boosted))
                learned.append(_ndcg(r, _corrected(r, model.predict(r["x"]))))
        print(
            f"leave-one-repo-out NDCG@10 over {len(base)} queries: hand-tuned {np.mean(base):.4f}, "
            f"learned {np.mean(learned):.4f} ({np.mean(learned) - np.mean(base):+.4f})",
            file=sys.stderr,
        )
        return

    model = _fit(records)
    payload = to_ensemble_dict(model.booster_)
    ensemble = TreeEnsemble.from_dict(payload)
    x = np.vstack([r["x"] for r in records])
    drift = float(np.abs(ensemble.predict(x) - model.predict(x)).max())
    if drift > 1e-6:
        raise RuntimeError(f"numpy evaluator disagrees with LightGBM by {drift:g}")
    args.output.write_text(json.dumps(payload, separators=(",", ":")) + "\n")
    print(f"wrote {args.output} ({len(payload['trees'])} trees, max |numpy - lightgbm| = {drift:.2g})", file=sys.stderr)


if __name__ == "__main__":
    main()
