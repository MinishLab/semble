import math
import random

_DEFAULT_SEED = 42


def bootstrap_ci(values: list[float], n_resamples: int = 10_000, seed: int = _DEFAULT_SEED) -> tuple[float, float]:
    """95% percentile bootstrap CI for the mean of ``values`` (stdlib-only, no scipy dep)."""
    if len(values) < 2:
        return (float("nan"), float("nan"))
    rng = random.Random(seed)
    n = len(values)
    means = []
    for _ in range(n_resamples):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    lo = means[int(0.025 * n_resamples)]
    hi = means[int(0.975 * n_resamples) - 1]
    return (lo, hi)


def mcnemar_exact_p(b: int, c: int) -> float:
    """Exact two-sided McNemar test on discordant pairs (``b`` vs ``c``), stdlib-only."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) * (0.5**n)
    return min(1.0, 2 * tail)
