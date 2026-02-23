"""Plotting utilities for benchmark visualization."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from .benchmark import BenchmarkResult


def _display_name(r: BenchmarkResult) -> str:
    """Generate a human-readable label for a benchmark result."""
    m_val = r.params.get("M")
    if r.method == "pq":
        return f"PQ(M={m_val or '?'})"
    if r.method == "pq+rerank":
        n_sl = r.params.get("n_shortlist", "")
        return f"PQ(M={m_val or '?'})+rerank({n_sl})"
    if r.method == "rabitq":
        return "RaBitQ"
    if r.method == "rabitq+rerank":
        n_sl = r.params.get("n_shortlist", "")
        return f"RaBitQ+rerank({n_sl})"
    return r.method


def setup_style():
    """Set up consistent plot style."""
    sns.set_theme(style="whitegrid", font_scale=1.1)
    plt.rcParams["figure.figsize"] = (10, 6)
    plt.rcParams["figure.dpi"] = 100


def plot_recall_vs_qps(
    results: list[BenchmarkResult],
    k: int = 10,
    save_path: Path | str | None = None,
    title: str | None = None,
):
    """Plot recall@k vs QPS Pareto curve.

    Args:
        results: List of benchmark results.
        k: Which recall@k to plot (uses recall_at_10 by default).
        save_path: Path to save figure. If None, shows interactively.
        title: Plot title.
    """
    setup_style()
    fig, ax = plt.subplots()

    method_styles = {
        "pq":             {"marker": "s", "color": "steelblue",  "label": "PQ"},
        "rabitq":         {"marker": "D", "color": "red",        "label": "RaBitQ"},
        "pq+rerank":      {"marker": "^", "color": "darkorange", "label": "PQ+rerank"},
        "rabitq+rerank":  {"marker": "v", "color": "darkviolet", "label": "RaBitQ+rerank"},
    }

    for method, style in method_styles.items():
        group = [r for r in results if r.method == method]
        if not group:
            continue
        recalls = [r.recall_at_10 for r in group]
        qps_vals = [r.qps for r in group]
        ax.scatter(recalls, qps_vals, marker=style["marker"], s=100,
                   color=style["color"], label=style["label"], zorder=5)
        for r, rec, qps in zip(group, recalls, qps_vals):
            m_val = r.params.get("M")
            if m_val is not None:
                ax.annotate(f"M={m_val}", (rec, qps), textcoords="offset points",
                            xytext=(5, 5), fontsize=9)

    ax.set_xlabel(f"Recall@{k}")
    ax.set_ylabel("Queries per Second (QPS)")
    ax.set_title(title or f"Recall@{k} vs QPS")
    ax.legend()

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
        print(f"Saved: {save_path}")
    else:
        plt.show()
    plt.close(fig)


def plot_memory_comparison(
    results: list[BenchmarkResult],
    save_path: Path | str | None = None,
    title: str | None = None,
):
    """Bar chart comparing memory usage across methods.

    Args:
        results: List of benchmark results.
        save_path: Path to save figure.
        title: Plot title.
    """
    setup_style()
    fig, ax = plt.subplots()

    names = []
    mem_kb = []
    colors = []

    color_map = {
        "pq": "steelblue",
        "rabitq": "indianred",
        "pq+rerank": "darkorange",
        "rabitq+rerank": "darkviolet",
    }
    for r in results:
        names.append(_display_name(r))
        colors.append(color_map.get(r.method, "gray"))
        mem_kb.append(r.memory_bytes / 1024)

    bars = ax.bar(names, mem_kb, color=colors)
    ax.set_ylabel("Memory (KB)")
    ax.set_title(title or "Memory Usage Comparison")

    for bar, val in zip(bars, mem_kb):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{val:.0f}KB", ha="center", va="bottom", fontsize=9)

    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
        print(f"Saved: {save_path}")
    else:
        plt.show()
    plt.close(fig)


def plot_error_histogram(
    results: list[BenchmarkResult],
    save_path: Path | str | None = None,
    title: str | None = None,
):
    """Histogram of distance estimation errors.

    Args:
        results: List of benchmark results.
        save_path: Path to save figure.
        title: Plot title.
    """
    setup_style()
    fig, ax = plt.subplots()

    for r in results:
        label = _display_name(r)
        # Show mean +/- std as a simple representation
        ax.errorbar(
            label, r.mean_distance_error,
            yerr=r.std_distance_error,
            fmt="o", capsize=5, capthick=2, markersize=8,
        )

    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax.set_ylabel("Mean Distance Estimation Error")
    ax.set_title(title or "Distance Estimation Bias")

    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
        print(f"Saved: {save_path}")
    else:
        plt.show()
    plt.close(fig)


def plot_error_vs_dimension(
    dimension_results: dict[int, list[BenchmarkResult]],
    save_path: Path | str | None = None,
):
    """Plot distance estimation error standard deviation vs dimensionality.

    Args:
        dimension_results: Mapping from dimension to results.
        save_path: Path to save figure.
    """
    setup_style()
    fig, ax = plt.subplots()

    dims = sorted(dimension_results.keys())

    # Collect PQ and RaBitQ errors by dimension
    pq_errors = {}
    rq_errors = {}
    for d in dims:
        for r in dimension_results[d]:
            if r.method == "rabitq":
                rq_errors[d] = r.std_distance_error
            elif r.method == "pq" and r.params.get("M") == 8:
                pq_errors[d] = r.std_distance_error

    if rq_errors:
        rq_dims = sorted(rq_errors.keys())
        ax.plot(rq_dims, [rq_errors[d] for d in rq_dims], "D-", color="red", label="RaBitQ")

    if pq_errors:
        pq_dims = sorted(pq_errors.keys())
        ax.plot(pq_dims, [pq_errors[d] for d in pq_dims], "s-", color="steelblue", label="PQ(M=8)")

    # Plot theoretical O(1/sqrt(D)) reference
    if rq_errors:
        d_ref = min(rq_errors.keys())
        scale = rq_errors[d_ref] * np.sqrt(d_ref)
        d_range = np.linspace(min(dims), max(dims), 100)
        ax.plot(d_range, scale / np.sqrt(d_range), "--", color="gray",
                alpha=0.5, label=r"$O(1/\sqrt{D})$ reference")

    ax.set_xlabel("Dimension (D)")
    ax.set_ylabel("Std of Distance Estimation Error")
    ax.set_title("Error Scaling with Dimension")
    ax.legend()

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
        print(f"Saved: {save_path}")
    else:
        plt.show()
    plt.close(fig)
