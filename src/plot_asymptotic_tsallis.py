"""Plotting utilities for asymptotic Tsallis/Shannon entropy estimator figures.

The script reads a simulation results table and exports the variance convergence
and asymptotic normality figures used in the study.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import norm


DISTRIBUTION_STYLES = {
    "GG_light": {"color": "#1f77b4", "marker": "o", "label": "GG (Light)"},
    "GG_heavy": {"color": "#d62728", "marker": "s", "label": "GG (Heavy)"},
    "Student": {"color": "#2ca02c", "marker": "^", "label": "Student-t"},
}

ESTIMATORS = ["Tsallis", "Shannon"]
DIMENSIONS_TO_COMPARE = [1, 5]


def configure_plot_style() -> None:
    """Set a consistent journal-style appearance for all figures."""
    plt.rcParams.update(
        {
            "text.usetex": False,
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "axes.labelsize": 12,
            "axes.titlesize": 13,
            "font.size": 11,
            "legend.fontsize": 10,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "lines.linewidth": 1.5,
            "axes.grid": True,
            "grid.alpha": 0.3,
            "grid.linestyle": "--",
        }
    )


def validate_columns(df: pd.DataFrame, required_columns: set[str]) -> None:
    """Check that the input table contains the variables required for plotting."""
    missing = required_columns.difference(df.columns)
    if missing:
        missing_list = ", ".join(sorted(missing))
        raise ValueError(f"The input table is missing required columns: {missing_list}")


def plot_variance_convergence(df: pd.DataFrame, output_dir: Path) -> None:
    """Create the variance convergence figure for Tsallis and Shannon estimators."""
    validate_columns(df, {"dist", "d", "estimator", "n", "var"})
    configure_plot_style()

    output_dir.mkdir(parents=True, exist_ok=True)
    sample_sizes = np.array(sorted(df["n"].dropna().unique()))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

    for panel_index, estimator in enumerate(ESTIMATORS):
        ax = axes[panel_index]
        estimator_df = df[df["estimator"] == estimator]

        for distribution, style in DISTRIBUTION_STYLES.items():
            for dimension in DIMENSIONS_TO_COMPARE:
                subset = estimator_df[
                    (estimator_df["dist"] == distribution) & (estimator_df["d"] == dimension)
                ].sort_values("n")

                if subset.empty:
                    continue

                line_style = "-" if dimension == 1 else "--"
                line_label = f"{style['label']}, d={dimension}"

                ax.loglog(
                    subset["n"],
                    subset["var"],
                    label=line_label,
                    color=style["color"],
                    marker=style["marker"],
                    linestyle=line_style,
                    markersize=7,
                    markerfacecolor="white",
                    markeredgewidth=1.5,
                )

        reference_subset = estimator_df[
            (estimator_df["dist"] == "Student") & (estimator_df["d"] == 5)
        ].sort_values("n")

        if reference_subset.empty:
            reference_subset = estimator_df.sort_values("n")

        if not reference_subset.empty and len(sample_sizes) > 0:
            first_row = reference_subset.iloc[0]
            reference_line = (first_row["var"] * first_row["n"]) / sample_sizes
            ax.loglog(
                sample_sizes,
                reference_line,
                color="black",
                linestyle=":",
                alpha=0.6,
                label=r"Theoretical $O(n^{-1})$",
            )

        ax.set_title(f" {estimator} Estimator", fontweight="bold")
        ax.set_xlabel(r"Sample Size ($n$)")
        if panel_index == 0:
            ax.set_ylabel(r"Variance ($\sigma^2$)")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, -0.05), ncol=4)

    plt.tight_layout()
    plt.savefig(output_dir / "variance_convergence.png", bbox_inches="tight", dpi=300)
    plt.close(fig)


def plot_asymptotic_normality(df: pd.DataFrame, output_dir: Path) -> None:
    """Create the standardized asymptotic normality figure."""
    validate_columns(df, {"n", "z", "estimator"})
    configure_plot_style()

    output_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(9, 6))

    sns.violinplot(
        data=df,
        x="n",
        y="z",
        hue="estimator",
        split=True,
        inner="quart",
        gap=0.1,
        palette={"Tsallis": "#1f77b4", "Shannon": "#a9a9a9"},
        alpha=0.8,
        ax=ax,
    )

    z_range = np.linspace(-4, 4, 100)
    density = norm.pdf(z_range, 0, 1)

    ax_inset = ax.inset_axes([0.88, 0.6, 0.1, 0.3])
    ax_inset.plot(density, z_range, color="red", lw=1.5, label="N(0,1)")
    ax_inset.fill_betweenx(z_range, density, color="red", alpha=0.1)
    ax_inset.set_title(r"Target: $\mathcal{N}(0,1)$", fontsize=9, color="red")
    ax_inset.axis("off")

    ax.axhline(0, color="black", lw=1, ls="-")
    ax.axhline(1.96, color="red", lw=1, ls="--", alpha=0.4)
    ax.axhline(-1.96, color="red", lw=1, ls="--", alpha=0.4)

    ax.set_xlabel(r"Sample Size ($n$)", fontweight="bold")
    ax.set_ylabel(r"Standardized Score ($z$)", fontweight="bold")
    sns.despine()
    ax.legend(title="Estimator", frameon=False, loc="upper left")

    plt.tight_layout()
    plt.savefig(output_dir / "asymptotic_normality_q1.png", bbox_inches="tight", dpi=300)
    plt.savefig(output_dir / "asymptotic_normality_q1.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate asymptotic Tsallis/Shannon estimator figures.")
    parser.add_argument("--input", type=Path, required=True, help="Path to the simulation results CSV file.")
    parser.add_argument("--output", type=Path, default=Path("figures"), help="Directory for exported figures.")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    plot_variance_convergence(df, args.output)
    plot_asymptotic_normality(df, args.output)


if __name__ == "__main__":
    main()
