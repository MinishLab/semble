import sys
from pathlib import Path
from typing import TypedDict

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

_RESULTS_DIR = Path(__file__).parent / "results"


class _Method(TypedDict):
    """Plot data for a single benchmark method."""

    name: str
    ndcg10: float
    index_ms: float
    query_p50_ms: float
    color: str
    params_m: float


_METHODS: list[_Method] = [
    {
        "name": "ripgrep",
        "ndcg10": 0.126,
        "index_ms": 0.0,
        "query_p50_ms": 12.08,
        "color": "#606060",
        "params_m": 0,
    },
    {
        "name": "BM25",
        "ndcg10": 0.673,
        "index_ms": 262.6,  # same semble index infrastructure; BM25 component adds negligible overhead
        "query_p50_ms": 0.019,
        "color": "#3a9e7e",
        "params_m": 0,
    },
    {
        "name": "ColGREP",
        "ndcg10": 0.6925,
        "index_ms": 5750.6,
        "query_p50_ms": 123.83,
        "color": "#e8a838",
        "params_m": 16,
    },
    {
        "name": "CodeRankEmbed",
        "ndcg10": 0.7648,
        "index_ms": 57269.4,
        "query_p50_ms": 16.27,
        "color": "#d9634f",
        "params_m": 137,
    },
    {
        "name": "CodeRankEmbed\nHybrid",
        "ndcg10": 0.8617,
        "index_ms": 57269.4,
        "query_p50_ms": 16.27,
        "color": "#922b21",
        "params_m": 137,
    },
    {
        "name": "semble",
        "ndcg10": 0.8544,
        "index_ms": 262.6,
        "query_p50_ms": 1.49,
        "color": "#1a5fa8",
        "params_m": 16,
    },
]

# Fixed label offset in cube-root(ms) space — gives a consistent visual gap at every x-position.
# The warm plot spans ~0.01–500 ms so needs a much smaller delta than the cold plot (~100 ms–100 s).
_CBRT_LABEL_DELTA_COLD = 2.0
_CBRT_LABEL_DELTA_WARM = 0.2

# Frontier methods per mode.
# Cold: incumbent prior-art curve (ripgrep → BM25 → ColGREP → CRE Hybrid); semble floats above it.
# Warm: BM25 dominates ripgrep (faster and higher NDCG), so incumbent curve is BM25 → CRE Hybrid.
_FRONTIER_NAMES: dict[str, set[str]] = {
    "cold": {"ripgrep", "BM25", "ColGREP", "CodeRankEmbed\nHybrid"},
    "warm": {"BM25", "CodeRankEmbed\nHybrid"},
}


def _marker_size(params_m: float) -> float:
    """Return scatter marker area scaling linearly with parameter count."""
    return max(80.0, 28.0 * params_m**0.5)


def _cbrt_forward(x: object) -> object:
    """Cube-root forward transform for x-axis scale."""
    return np.cbrt(x)  # type: ignore[call-overload]


def _cbrt_inverse(x: object) -> object:
    """Cube-root inverse transform for x-axis scale."""
    return np.power(x, 3)  # type: ignore[call-overload]


def _format_ms(v: float, _: object) -> str:
    """Format milliseconds as a human-readable time string."""
    if v >= 1_000:
        return f"{v / 1_000:.0f} s"
    if v < 1:
        return f"{v:.1g} ms"
    return f"{v:.0f} ms"


def _make_plot(out_path: Path, *, warm: bool = False) -> None:
    """Generate a speed-vs-quality scatter plot.

    :param out_path: Destination PNG path.
    :param warm: If True, use per-query latency (index pre-built). If False, use index + query latency.
    """
    mode = "warm" if warm else "cold"
    cbrt_label_delta = _CBRT_LABEL_DELTA_WARM if warm else _CBRT_LABEL_DELTA_COLD

    fig, ax = plt.subplots(figsize=(8, 5))

    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    ax.grid(axis="y", color="#e8e8e8", linewidth=0.7, zorder=0)
    ax.grid(axis="x", color="#f0f0f0", linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)

    # Remove top and right spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#cccccc")
    ax.spines["bottom"].set_color("#cccccc")

    # Frontier line connecting the incumbent methods in speed order.
    frontier = sorted(
        [
            (m["query_p50_ms"] if warm else m["index_ms"] + m["query_p50_ms"], m["ndcg10"])
            for m in _METHODS
            if m["name"] in _FRONTIER_NAMES[mode]
        ]
    )
    xlim = (0.01, 500) if warm else (5, 200_000)
    # Shade the incumbent zone: region below the frontier, closed to the plot edges.
    shade_xs = [xlim[0]] + [p[0] for p in frontier] + [xlim[1]]
    shade_ys = [frontier[0][1]] + [p[1] for p in frontier] + [frontier[-1][1]]
    ax.fill_between(shade_xs, shade_ys, 0.0, color="#1a5fa8", alpha=0.07, zorder=0.5, linewidth=0)
    ax.plot(
        [p[0] for p in frontier],
        [p[1] for p in frontier],
        color="#cccccc",
        linewidth=1.0,
        linestyle="--",
        zorder=1,
    )

    for m in _METHODS:
        x = m["query_p50_ms"] if warm else m["index_ms"] + m["query_p50_ms"]
        y = m["ndcg10"]
        ax.scatter(
            x,
            y,
            s=_marker_size(m["params_m"]),
            color=m["color"],
            marker="o",
            zorder=3,
            linewidths=1.2,
            edgecolors="white",
        )

        x_label = (x ** (1 / 3) + cbrt_label_delta) ** 3
        ax.text(x_label, y, m["name"], fontsize=8.5, color=m["color"], ha="left", va="center", zorder=4)

    ax.set_xscale("function", functions=(_cbrt_forward, _cbrt_inverse))
    ax.set_ylabel("NDCG@10", fontsize=10, color="#444444")
    ax.set_ylim(0.05, 0.95)

    if warm:
        ax.set_xlabel("Query latency (warm)", fontsize=10, color="#444444")
        ax.set_xlim(0.01, 500)
        ax.set_xticks([0.1, 1, 10, 100])
    else:
        ax.set_xlabel("Query latency (cold)", fontsize=10, color="#444444")
        ax.set_xlim(5, 200_000)
        ax.set_xticks([100, 1_000, 10_000, 100_000])

    ax.xaxis.set_major_formatter(ticker.FuncFormatter(_format_ms))
    ax.tick_params(labelsize=9, colors="#555555")

    # Model-size legend
    legend_entries = [(0, "no model"), (16, "16 M params"), (137, "137 M params")]
    handles = [
        plt.scatter(
            [],
            [],
            s=_marker_size(p),
            color="#aaaaaa",
            marker="o",
            label=label,
            edgecolors="white",
            linewidths=1.2,
        )
        for p, label in legend_entries
    ]
    legend = ax.legend(
        handles=handles,
        title="Model size",
        fontsize=8.5,
        title_fontsize=9,
        loc="lower right",
        frameon=True,
        framealpha=0.95,
        edgecolor="#dddddd",
        labelspacing=1.2,
        borderpad=1.5,
    )
    legend.get_title().set_color("#444444")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved to {out_path}", file=sys.stderr)


def main() -> None:
    """Generate cold and warm speed-vs-quality scatter plots."""
    _make_plot(_RESULTS_DIR / "speed_vs_ndcg_cold.png")
    _make_plot(_RESULTS_DIR / "speed_vs_ndcg_warm.png", warm=True)


if __name__ == "__main__":
    main()
