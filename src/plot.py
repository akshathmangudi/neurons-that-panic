"""
Visualizations
"""

import os

import matplotlib.pyplot as plt
import seaborn as sns


def setup_style():
    """Configure matplotlib/seaborn style."""
    sns.set_style("whitegrid")
    plt.rcParams["figure.figsize"] = (10, 6)
    plt.rcParams["font.size"] = 12
    plt.rcParams["axes.labelsize"] = 12
    plt.rcParams["axes.titlesize"] = 14
    plt.rcParams["xtick.labelsize"] = 10
    plt.rcParams["ytick.labelsize"] = 10
    plt.rcParams["legend.fontsize"] = 10
    plt.rcParams["figure.dpi"] = 300


def layerwise(df, out_path):
    """Save layerwise mean |Δnorm| plot."""
    layerwise_stats = df.groupby("layer")["abs_delta_norm"].mean().sort_index()

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(
        layerwise_stats.index,
        layerwise_stats.values,
        color="steelblue",
        alpha=0.7,
        edgecolor="black",
        linewidth=1.2,
    )
    ax.set_xlabel("Layer Index", fontsize=12, fontweight="bold")
    ax.set_ylabel("Mean |Δnorm|", fontsize=12, fontweight="bold")
    ax.set_title(
        "Layerwise Mean |Δnorm| Across All Components", fontsize=14, fontweight="bold"
    )
    ax.set_xticks(layerwise_stats.index)
    ax.grid(axis="y", alpha=0.3, linestyle="--")

    for i, (layer, value) in enumerate(layerwise_stats.items()):
        ax.text(layer, value, f"{value:.2f}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


def positional(df, top_k, out_path):
    """Save positional localization histogram."""
    top_components = df.head(top_k)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(
        top_components["abs_delta_norm"],
        bins=20,
        color="coral",
        alpha=0.7,
        edgecolor="black",
        linewidth=1.2,
    )
    ax.set_xlabel("|Δnorm| at Trigger Position", fontsize=12, fontweight="bold")
    ax.set_ylabel("Number of Components", fontsize=12, fontweight="bold")
    ax.set_title(
        f"Distribution of |Δnorm| for Top-{top_k} Panic Components",
        fontsize=14,
        fontweight="bold",
    )
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.axvline(
        top_components["abs_delta_norm"].mean(),
        color="red",
        linestyle="--",
        linewidth=2,
        label=f'Mean: {top_components["abs_delta_norm"].mean():.2f}',
    )
    ax.legend()

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


def distributions(df, out_path):
    """Save MLP vs Attention |Δnorm| distributions."""
    mlp_delta_norm = df[df["type"] == "mlp"]["abs_delta_norm"]
    attn_delta_norm = df[df["type"] == "attn"]["abs_delta_norm"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    ax1.hist(
        mlp_delta_norm,
        bins=50,
        alpha=0.6,
        label="MLP",
        color="steelblue",
        edgecolor="black",
        linewidth=0.5,
    )
    ax1.hist(
        attn_delta_norm,
        bins=50,
        alpha=0.6,
        label="Attention",
        color="coral",
        edgecolor="black",
        linewidth=0.5,
    )
    ax1.set_xlabel("|Δnorm|", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Number of Components", fontsize=12, fontweight="bold")
    ax1.set_title(
        "Distribution of |Δnorm|: MLP vs Attention", fontsize=14, fontweight="bold"
    )
    ax1.legend()
    ax1.grid(axis="y", alpha=0.3, linestyle="--")

    box_data = [mlp_delta_norm.values, attn_delta_norm.values]
    ax2.boxplot(
        box_data,
        labels=["MLP", "Attention"],
        patch_artist=True,
        boxprops=dict(facecolor="lightblue", alpha=0.7),
        medianprops=dict(color="red", linewidth=2),
    )
    ax2.set_ylabel("|Δnorm|", fontsize=12, fontweight="bold")
    ax2.set_title("Box Plot: MLP vs Attention |Δnorm|", fontsize=14, fontweight="bold")
    ax2.grid(axis="y", alpha=0.3, linestyle="--")

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


def patching(results, out_path):
    """Save causal patching bar plot."""
    top_k_recovery = results["mean_recovery"]
    random_k_mean = results["mean_random_recovery"]
    random_k_std = results["std_random_recovery"]

    fig, ax = plt.subplots(figsize=(8, 6))
    categories = ["Top-K\nComponents", "Random-K\nBaseline"]
    values = [top_k_recovery, random_k_mean]
    errors = [0, random_k_std]
    colors = ["steelblue", "coral"]

    ax.bar(
        categories,
        values,
        yerr=errors,
        capsize=10,
        color=colors,
        alpha=0.7,
        edgecolor="black",
        linewidth=1.5,
        error_kw={"elinewidth": 2, "capthick": 2},
    )

    for i, (val, err) in enumerate(zip(values, errors)):
        if i == 0:
            ax.text(
                i,
                val,
                f"{val:.3f}",
                ha="center",
                va="bottom",
                fontsize=11,
                fontweight="bold",
            )
        else:
            ax.text(
                i,
                val + err,
                f"{val:.3f}±{err:.3f}",
                ha="center",
                va="bottom",
                fontsize=11,
                fontweight="bold",
            )

    ax.set_ylabel("Recovery Metric", fontsize=12, fontweight="bold")
    ax.set_title(
        "Causal Patching Results: Top-K vs Random-K Baseline",
        fontsize=14,
        fontweight="bold",
    )
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.set_ylim(bottom=0)
    ax.axhline(y=0, color="black", linestyle="-", linewidth=0.5)

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
