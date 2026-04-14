"""Diverse top-k selection with file-path penalties and saturation decay."""

from semble.ranking.penalties import _file_path_penalty, _is_test_file
from semble.types import Chunk

# Maximum chunks from the same file before a saturation penalty is applied.
_FILE_SATURATION_THRESHOLD = 2

# Multiplicative penalty per extra chunk from the same file beyond the threshold.
_FILE_SATURATION_DECAY = 0.5


def diverse_topk(
    scores: dict[Chunk, float],
    top_k: int,
) -> list[tuple[Chunk, float]]:
    """Select top-k results with file-path penalties and file-saturation decay.

    File-path penalties (test files, init files, compat dirs, etc.) are applied
    first.  Then candidates are processed in descending penalised-score order
    with file-saturation decay applied greedily.  Because decay only reduces
    scores and candidates are sorted by penalised score descending, we can stop
    early once the remaining penalised scores cannot beat the current top-k floor.

    :param scores: Combined scores for all candidate chunks.
    :param top_k: Number of results to return.
    :return: Selected (chunk, effective_score) pairs in descending effective-score order.
    """
    if not scores:
        return []

    # Apply file-path penalties.
    penalty_cache: dict[str, float] = {}
    penalised: dict[Chunk, float] = {}
    for chunk, score in scores.items():
        if chunk.file_path not in penalty_cache:
            is_test = _is_test_file(chunk.file_path)
            penalty_cache[chunk.file_path] = _file_path_penalty(chunk.file_path, is_test=is_test)
        penalised[chunk] = score * penalty_cache[chunk.file_path]

    # Sort by penalised score (highest first) — single sort.
    ranked = sorted(penalised, key=lambda c: -penalised[c])

    # Greedy pass with file-saturation decay and early-exit.
    # Candidates are already sorted by pen_score descending, so pen_score is
    # an upper bound on any future eff_score (decay only reduces scores).
    # Once we have top_k items, any candidate whose pen_score cannot beat the
    # current k-th best effective score can be skipped — and so can every
    # subsequent candidate, so we break.
    # min_selected tracks the minimum effective score among the top_k collected
    # so far; it is recomputed after each addition to stay accurate.
    file_selected: dict[str, int] = {}
    selected: list[tuple[float, Chunk]] = []
    min_selected = float("+inf")

    for chunk in ranked:
        pen_score = penalised[chunk]

        if len(selected) >= top_k and pen_score <= min_selected:
            break

        already_selected = file_selected.get(chunk.file_path, 0)
        eff_score = pen_score
        if already_selected >= _FILE_SATURATION_THRESHOLD:
            excess = already_selected - _FILE_SATURATION_THRESHOLD + 1
            eff_score *= _FILE_SATURATION_DECAY**excess

        selected.append((eff_score, chunk))
        file_selected[chunk.file_path] = already_selected + 1

        if len(selected) >= top_k:
            min_selected = min(s for s, _ in selected)

    selected.sort(key=lambda t: -t[0])
    return [(chunk, score) for score, chunk in selected[:top_k]]
