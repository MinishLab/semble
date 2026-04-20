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
        "ndcg10": 0.123,
        "index_ms": 0.0,
        "query_p50_ms": 12.08,
        "color": "#606060",
        "params_m": 0,
    },
    {
        "name": "colgrep",
        "ndcg10": 0.577,
        "index_ms": 5750.6,
        "query_p50_ms": 123.83,
        "color": "#e8a838",
        "params_m": 16,
    },
    {
        "name": "coderankembed\nsemantic",
        "ndcg10": 0.762,
        "index_ms": 57269.4,
        "query_p50_ms": 16.27,
        "color": "#d9634f",
        "params_m": 137,
    },
    {
        "name": "coderankembed\nhybrid",
        "ndcg10": 0.860,
        "index_ms": 57269.4,
        "query_p50_ms": 16.27,
        "color": "#922b21",
        "params_m": 137,
    },
    {
        "name": "semble",
        "ndcg10": 0.852,
        "index_ms": 262.6,
        "query_p50_ms": 1.49,
        "color": "#1a5fa8",
        "params_m": 16,
    },
]

# Fixed label offset in cube-root(ms) space — gives a consistent visual gap at every x-position.
_CBRT_LABEL_DELTA = 2.0


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
    return f"{v:.0f} ms"


def _make_plot(out_path: Path) -> None:
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

    for m in _METHODS:
        x = m["index_ms"] + m["query_p50_ms"]
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

        x_label = (x ** (1 / 3) + _CBRT_LABEL_DELTA) ** 3
        ax.text(x_label, y, m["name"], fontsize=8.5, color=m["color"], ha="left", va="center", zorder=4)

    ax.set_xscale("function", functions=(_cbrt_forward, _cbrt_inverse))
    ax.set_xlabel("Time to first result — index + query", fontsize=10, color="#444444")
    ax.set_ylabel("NDCG@10", fontsize=10, color="#444444")
    ax.set_xlim(5, 200_000)
    ax.set_ylim(0.05, 0.95)

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
    """Generate the speed-vs-quality scatter plot."""
    out = _RESULTS_DIR / "speed_vs_ndcg.png"
    _make_plot(out)


if __name__ == "__main__":
    main()
