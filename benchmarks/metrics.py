import math

from benchmarks.data import Target, path_matches, target_matches_location
from semble.types import SearchResult


def dcg(relevances: list[int]) -> float:
    """Compute Discounted Cumulative Gain for a ranked relevance list."""
    return sum(rel / math.log2(i + 2) for i, rel in enumerate(relevances))


def ndcg_at_k(relevant_ranks: list[int], n_relevant: int, k: int) -> float:
    """Compute NDCG@k given 1-based ranks of relevant results and total relevant count."""
    if n_relevant == 0:
        return 0.0
    relevances = [0] * k
    for rank in relevant_ranks:
        if 1 <= rank <= k:
            relevances[rank - 1] = 1
    ideal = dcg([1] * min(k, n_relevant))
    return dcg(relevances) / ideal if ideal > 0 else 0.0


def target_rank(results: list[SearchResult], target: Target) -> int | None:
    """Return 1-based rank of the first chunk result covering target, or None."""
    for index, result in enumerate(results, 1):
        chunk = result.chunk
        if target_matches_location(chunk.file_path, chunk.start_line, chunk.end_line, target):
            return index
    return None


def file_rank(file_paths: list[str], target_path: str) -> int | None:
    """Return 1-based rank of the first file path matching target_path, or None."""
    for i, fp in enumerate(file_paths, 1):
        if path_matches(fp, target_path):
            return i
    return None
