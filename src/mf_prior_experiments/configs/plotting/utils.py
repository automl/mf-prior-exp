from __future__ import annotations

from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Any
from warnings import warn

import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats

from .styles import ALGORITHMS, BENCHMARK_COLORS, COLOR_MARKER_DICT, DATASETS


def parse_args() -> Namespace:
    parser = ArgumentParser(description="mf-prior-exp plotting")

    parser.add_argument("--filename", type=str, required=True)

    parser.add_argument("--experiment_group", type=str, required=True)
    parser.add_argument("--algorithms", nargs="+", required=True)

    parser.add_argument("--benchmarks", nargs="+", default=None)

    parser.add_argument("--relative_rankings", action="store_true")
    parser.add_argument("--benchmarks1", nargs="+", default=None)
    parser.add_argument("--benchmarks2", nargs="+", default=None)
    parser.add_argument("--benchmarks3", nargs="+", default=None)
    parser.add_argument("--benchmarks4", nargs="+", default=None)

    parser.add_argument("--base_path", type=Path, default=None)
    parser.add_argument("--n_workers", type=int, default=1)
    parser.add_argument("--budget", nargs="+", type=float, default=None)
    parser.add_argument("--x_range", nargs=2, type=float, default=None)

    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument("--ext", type=str, choices=["pdf", "png"], default="pdf")
    parser.add_argument("--plot_default", action="store_true")
    parser.add_argument("--plot_optimum", action="store_true")
    parser.add_argument("--dynamic_y_lim", action="store_true")
    parser.add_argument("--parallel", action="store_true")

    args = parser.parse_args()

    if args.budget:
        raise ValueError("CD plots (which use --budget) not supported yet")

    if args.relative_rankings:
        benches = [args.benchmarks1, args.benchmarks2, args.benchmarks3, args.benchmarks4]
        if any(b is None for b in benches):
            raise ValueError("Must specify all benchmarks{1,2,3,4} for relative rankings")

        total_benchmarks = [*args.benchmarks1, *args.benchmarks2, *args.benchmarks3, *args.benchmarks4,]
        if not len(total_benchmarks) == len(set(total_benchmarks)):
            msg = (
                "Benchmarks in benchmarks{1,2,3,4} must all be unique\n"
                f"--benchmarks1 {args.benchmarks1}\n"
                f"--benchmarks2 {args.benchmarks2}\n"
                f"--benchmarks3 {args.benchmarks3}\n"
                f"--benchmarks4 {args.benchmarks4}"
            )
            raise ValueError(msg)
    else:
        if args.benchmarks is None:
            raise ValueError("Must specify --benchmarks")

    return args

def set_general_plot_style():
    """
    sns.set_style("ticks")
    sns.set_context("paper")
    sns.set_palette("deep")
    """
    # plt.switch_backend("pgf")
    plt.rcParams.update(
        {
            "text.usetex": False,  # True,
            # "pgf.texsystem": "pdflatex",
            # "pgf.rcfonts": False,
            # "font.family": "serif",
            # "font.serif": [],
            # "font.sans-serif": [],
            # "font.monospace": [],
            "font.size": "10.90",
            "legend.fontsize": "9.90",
            "xtick.labelsize": "small",
            "ytick.labelsize": "small",
            "legend.title_fontsize": "small",
            # "bottomlabel.weight": "normal",
            # "toplabel.weight": "normal",
            # "leftlabel.weight": "normal",
            # "tick.labelweight": "normal",
            # "title.weight": "normal",
            # "pgf.preamble": r"""
            #    \usepackage[T1]{fontenc}
            #    \usepackage[utf8x]{inputenc}
            #    \usepackage{microtype}
            # """,
        }
    )

def plot_incumbent(
    ax: plt.Axes,
    df: pd.DataFrame,
    algorithm: str,
    xlabel: str | None = None,
    ylabel: str | None = None,
    title: str | None = None,
    x_range: tuple[float, float] | None = None,
    plot_default: float | None = None,
    plot_optimum: float | None = None,
    force_prior_line: bool = False,
    **plot_kwargs: Any,
):
    x = df.index
    y_mean = df.mean(axis=1).values
    std_error = stats.sem(df.values, axis=1)

    ax.step(
        x,
        y_mean,
        label=ALGORITHMS[algorithm],
        **plot_kwargs,
        color=COLOR_MARKER_DICT[algorithm],
        linestyle="-" if "prior" in algorithm else "-",
        linewidth=1,
        where="post",
    )
    if force_prior_line or (plot_default is not None and plot_default < y_mean[0]):
        # plot only if the default score is better than the first incumbent plotted
        ax.plot(
            x,
            [plot_default] * len(x),
            color="black",
            linestyle=":",
            linewidth=1.0,
            dashes=(5, 10),
            label="Mode",
        )

    if plot_optimum is not None and plot_optimum < y_mean[0]:
        # plot only if the optimum score is better than the first incumbent plotted
        ax.plot(
            x,
            [plot_optimum] * len(x),
            color="black",
            linestyle="-.",
            linewidth=1.2,
            label="Optimum",
        )

    default_color_marker = "black"
    if algorithm not in COLOR_MARKER_DICT:
        warn(
            f"Could not find color for algorithm {algorithm}, using {default_color_marker}"
        )

    ax.fill_between(
        x,
        y_mean - std_error,
        y_mean + std_error,
        color=COLOR_MARKER_DICT.get(algorithm, default_color_marker),
        alpha=0.1,
        step="post",
    )

    ax.set_xlim(auto=True)

    if title is not None:
        default_color = "black"
        if title not in BENCHMARK_COLORS:
            warn(f"Could not find color for benchmark {title}, using {default_color}")
        if title not in DATASETS:
            warn(f"Could not find title for benchmark {title}, using {title}")

        ax.set_title(
            DATASETS.get(title, title),
            fontsize=20,
            color=BENCHMARK_COLORS.get(title, default_color),
        )

    if xlabel is not None:
        ax.set_xlabel(xlabel, fontsize=18, color=(0, 0, 0, 0.69))

    if ylabel is not None:
        ax.set_ylabel(ylabel, fontsize=18, color=(0, 0, 0, 0.69))

    if x_range is not None:
        ax.set_xlim(*x_range)
        if x_range == [1, 12]:
            ax.set_xticks([1, 3, 5, 10, 12], [1, 3, 5, 10, 12])  # type: ignore

    # Black with some alpha
    ax.tick_params(axis="both", which="major", labelsize=18, labelcolor=(0, 0, 0, 0.69))
    ax.grid(True, which="both", ls="-", alpha=0.8)
