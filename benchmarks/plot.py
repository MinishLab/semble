import json
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
    marker: str
    params_m: float


_METHODS: list[_Method] = [
    {
        "name": "ripgrep\n(no index)",
        "ndcg10": 0.123,
        "index_ms": 0.0,
        "query_p50_ms": 12.08,
        "color": "#888888",
        "marker": "s",
        "params_m": 0,
    },
    {
        "name": "colgrep",
        "ndcg10": 0.577,
        "index_ms": 5750.6,
        "query_p50_ms": 123.83,
        "color": "#e07b39",
        "marker": "D",
        "params_m": 0,
    },
    {
        "name": "coderankembed\nsemantic",
        "ndcg10": 0.762,
        "index_ms": 57269.4,
        "query_p50_ms": 16.27,
        "color": "#c0392b",
        "marker": "^",
        "params_m": 137,
    },
    {
        "name": "coderankembed\nhybrid",
        "ndcg10": 0.860,
        "index_ms": 57269.4,
        "query_p50_ms": 16.27,
        "color": "#922b21",
        "marker": "v",
        "params_m": 137,
    },
    {
        "name": "semble",
        "ndcg10": 0.852,
        "index_ms": 262.6,
        "query_p50_ms": 1.49,
        "color": "#2471a3",
        "marker": "o",
        "params_m": 16,
    },
]

# Label offsets (x_factor, y_offset) to avoid overlaps
_OFFSETS: dict[str, tuple[float, float]] = {
    "ripgrep\n(no index)": (1.3, -0.03),
    "colgrep": (1.3, 0.008),
    "coderankembed\nsemantic": (0.3, -0.048),
    "coderankembed\nhybrid": (0.3, 0.018),
    "semble": (1.3, 0.008),
}


def _make_plot(out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))

    ax.set_facecolor("#fafafa")
    fig.patch.set_facecolor("white")
    ax.grid(True, which="both", color="#e0e0e0", linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)

    # Marker size scales with sqrt(params) so the area scales linearly with params.
    # Floor at 60 so zero-param tools are still visible.
    def _marker_size(params_m: float) -> float:
        return max(60.0, 18.0 * params_m**0.5)

    for m in _METHODS:
        x = m["index_ms"] + m["query_p50_ms"]
        y = m["ndcg10"]
        ax.scatter(
            x,
            y,
            s=_marker_size(m["params_m"]),
            color=m["color"],
            marker=m["marker"],
            zorder=3,
            linewidths=0.8,
            edgecolors="white",
        )
        xf, yo = _OFFSETS.get(m["name"], (1.3, 0.0))
        ax.annotate(
            m["name"],
            xy=(x, y),
            xytext=(x * xf, y + yo),
            fontsize=8.5,
            color=m["color"],
            va="center",
            zorder=4,
        )

    ax.set_xscale("log")
    ax.set_xlabel("Time to first result (index + query, ms)", fontsize=10)
    ax.set_ylabel("NDCG@10", fontsize=10)
    ax.set_xlim(5, 200_000)
    ax.set_ylim(0.05, 0.95)

    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v:,.0f} ms"))
    ax.tick_params(labelsize=8.5)

    # Legend for marker sizes
    legend_params = [(0, "no model"), (16, "16M params"), (137, "137M params")]
    handles = [
        plt.scatter(
            [],
            [],
            s=_marker_size(p),
            color="#999999",
            marker="o",
            label=label,
            edgecolors="white",
            linewidths=0.8,
        )
        for p, label in legend_params
    ]
    ax.legend(handles=handles, title="Model size", fontsize=8, title_fontsize=8.5, loc="lower right", framealpha=0.9)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved to {out_path}", file=__import__("sys").stderr)


def main() -> None:
    """Generate the speed-vs-quality scatter plot."""
    out = _RESULTS_DIR / "speed_vs_ndcg.png"
    _make_plot(out)


if __name__ == "__main__":
    main()
