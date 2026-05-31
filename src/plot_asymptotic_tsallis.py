from pathlib import Path
import zipfile
import textwrap

base = Path("/mnt/data/asymptotic_tsallis_complete_code")
src = base / "src"
data = base / "data"
figures = base / "figures"

for folder in [src, data, figures]:
    folder.mkdir(parents=True, exist_ok=True)

script = r'''"""Figure generation utilities for asymptotic Tsallis and Shannon entropy estimators.

This script reads a simulation-results table and exports the figures used to
summarise variance convergence and asymptotic normality.

Expected input columns
----------------------
Variance convergence figure:
    dist, d, estimator, n, var

Asymptotic normality figure:
    n, z, estimator

Optional columns:
    q
        If present, the script can filter the table by the selected q value.

Example
-------
python src/plot_asymptotic_tsallis.py \
    --input data/asymptotic_tsallis_results.csv \
    --output figures \
    --q 1.0
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

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
    """Apply a consistent journal-style appearance to all figures."""
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


def validate_columns(df: pd.DataFrame, required_columns: Iterable[str]) -> None:
    """Raise a clear error if the input table does not contain required columns."""
    required = set(required_columns)
    missing = required.difference(df.columns)

    if missing:
        missing_list = ", ".join(sorted(missing))
        available_list = ", ".join(df.columns)
        raise ValueError(
            "The input table is missing required columns: "
            f"{missing_list}. Available columns are: {available_list}"
        )


def normalise_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Standardise common column-name variants before plotting."""
    df = df.copy()
    df.columns = [str(col).strip() for col in df.columns]

    aliases = {
        "distribution": "dist",
        "dimension": "d",
        "sample_size": "n",
        "sample size": "n",
        "variance": "var",
        "standardized_score": "z",
        "standardized score": "z",
        "z_score": "z",
        "z-score": "z",
    }

    rename_map = {}
    for column in df.columns:
        key = column.lower().strip()
        if key in aliases:
            rename_map[column] = aliases[key]

    return df.rename(columns=rename_map)


def filter_by_q(df: pd.DataFrame, q_value: float | None) -> pd.DataFrame:
    """Filter the data by q when a q column is available."""
    if q_value is None or "q" not in df.columns:
        return df

    filtered = df[np.isclose(df["q"].astype(float), q_value)]

    if filtered.empty:
        raise ValueError(f"No rows were found for q={q_value}.")

    return filtered


def save_figure(fig: plt.Figure, output_dir: Path, filename: str) -> None:
    """Save a figure in PNG and PDF formats."""
    output_dir.mkdir(parents=True, exist_ok=True)

    fig.savefig(output_dir / f"{filename}.png", bbox_inches="tight", dpi=300)
    fig.savefig(output_dir / f"{filename}.pdf", bbox_inches="tight")


def collect_unique_legend_entries(axes: Iterable[plt.Axes]) -> tuple[list, list]:
    """Collect unique legend entries from one or more axes."""
    handles_by_label = {}

    for ax in axes:
        handles, labels = ax.get_legend_handles_labels()
        for handle, label in zip(handles, labels):
            if label and label not in handles_by_label:
                handles_by_label[label] = handle

    return list(handles_by_label.values()), list(handles_by_label.keys())


def plot_variance_convergence(df: pd.DataFrame, output_dir: Path) -> None:
    """Create the variance convergence figure for Tsallis and Shannon estimators."""
    validate_columns(df, {"dist", "d", "estimator", "n", "var"})
    configure_plot_style()

    plot_df = df.copy()
    plot_df["d"] = plot_df["d"].astype(int)
    plot_df["n"] = plot_df["n"].astype(float)
    plot_df["var"] = plot_df["var"].astype(float)

    sample_sizes = np.array(sorted(plot_df["n"].dropna().unique()))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

    for panel_index, estimator in enumerate(ESTIMATORS):
        ax = axes[panel_index]
        estimator_df = plot_df[plot_df["estimator"] == estimator]

        for distribution, style in DISTRIBUTION_STYLES.items():
            for dimension in DIMENSIONS_TO_COMPARE:
                subset = estimator_df[
                    (estimator_df["dist"] == distribution)
                    & (estimator_df["d"] == dimension)
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

        ax.set_title(f"{estimator} Estimator", fontweight="bold")
        ax.set_xlabel(r"Sample Size ($n$)")

        if panel_index == 0:
            ax.set_ylabel(r"Variance ($\sigma^2$)")

    handles, labels = collect_unique_legend_entries(axes)
    if handles:
        fig.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.05),
            ncol=4,
            frameon=False,
        )

    fig.tight_layout()
    save_figure(fig, output_dir, "variance_convergence")
    plt.close(fig)


def plot_asymptotic_normality(df: pd.DataFrame, output_dir: Path) -> None:
    """Create the standardized asymptotic normality figure."""
    validate_columns(df, {"n", "z", "estimator"})
    configure_plot_style()

    plot_df = df.copy()
    plot_df["n"] = plot_df["n"].astype(str)
    plot_df["z"] = plot_df["z"].astype(float)

    estimator_count = plot_df["estimator"].nunique()
    use_split_violin = estimator_count == 2

    fig, ax = plt.subplots(figsize=(9, 6))

    violin_kwargs = {
        "data": plot_df,
        "x": "n",
        "y": "z",
        "hue": "estimator",
        "inner": "quart",
        "palette": {"Tsallis": "#1f77b4", "Shannon": "#a9a9a9"},
        "alpha": 0.8,
        "ax": ax,
    }

    if use_split_violin:
        violin_kwargs["split"] = True

    try:
        sns.violinplot(**violin_kwargs, gap=0.1)
    except TypeError:
        sns.violinplot(**violin_kwargs)

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

    fig.tight_layout()
    save_figure(fig, output_dir, "asymptotic_normality_q1")
    plt.close(fig)


def load_results(input_path: Path, q_value: float | None) -> pd.DataFrame:
    """Load and prepare the simulation-results table."""
    if not input_path.exists():
        raise FileNotFoundError(
            f"The input file was not found: {input_path}. "
            "Place the CSV file in the data folder or pass the correct path using --input."
        )

    df = pd.read_csv(input_path)
    df = normalise_column_names(df)
    df = filter_by_q(df, q_value)

    return df


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate asymptotic Tsallis/Shannon entropy estimator figures."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/asymptotic_tsallis_results.csv"),
        help="Path to the simulation-results CSV file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("figures"),
        help="Directory where figures will be exported.",
    )
    parser.add_argument(
        "--q",
        type=float,
        default=1.0,
        help="Optional q value used when the input table contains a q column.",
    )
    parser.add_argument(
        "--skip-variance",
        action="store_true",
        help="Do not generate the variance convergence figure.",
    )
    parser.add_argument(
        "--skip-normality",
        action="store_true",
        help="Do not generate the asymptotic normality figure.",
    )

    args = parser.parse_args()
    df = load_results(args.input, args.q)

    if not args.skip_variance:
        plot_variance_convergence(df, args.output)

    if not args.skip_normality:
        plot_asymptotic_normality(df, args.output)

    print(f"Figures were exported to: {args.output.resolve()}")


if __name__ == "__main__":
    main()
'''

readme = """# Asymptotic Tsallis Entropy Estimator Figures

This repository contains the plotting code used to reproduce the asymptotic-analysis figures for Tsallis and Shannon entropy estimators.

## Repository structure

```text
.
├── README.md
├── requirements.txt
├── data/
│   └── README.md
├── figures/
│   └── .gitkeep
└── src/
    └── plot_asymptotic_tsallis.py
