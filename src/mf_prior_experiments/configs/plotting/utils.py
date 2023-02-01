import argparse

import matplotlib.pyplot as plt
import mfpbench
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats

from .styles import ALGORITHMS, BENCHMARK_COLORS, COLOR_MARKER_DICT, DATASETS

def get_max_fidelity(benchmark_name):
    if "lcbench" in benchmark_name:
        name, task_id, _ = benchmark_name.split("-")
        task_id = task_id.replace("_prior", "")
        bench = mfpbench.get(name, task_id=task_id)
    else:
        benchmark_name, _ = benchmark_name.split("-")
        benchmark_name = benchmark_name.replace("_prior", "")
        bench = mfpbench.get(benchmark_name)
    _, upper, _ = bench.fidelity_range

    return upper


def get_parser():
    parser = argparse.ArgumentParser(
        description="mf-prior-exp plotting",
    )
    parser.add_argument(
        "--base_path",
        type=Path,
        default=None,
        help="path where `results/` exists",
    )
    parser.add_argument("--experiment_group", type=str, default="")
    parser.add_argument(
        "--n_workers",
        type=int,
        default=1,
        help="for multiple workers we plot based on end timestamps on "
        "x-axis (no continuation considered); any value > 1 is adequate",
    )
    parser.add_argument("--benchmarks", nargs="+", default=None)
    parser.add_argument("--algorithms", nargs="+", default=None)
    parser.add_argument("--plot_id", type=str, default="1")
    parser.add_argument("--research_question", type=int, default=1)
    parser.add_argument(
        "--which_prior",
        type=str,
        choices=["good", "bad"],
        default="bad",
        help="for RQ2 choose whether to plot good or bad",
    )
    parser.add_argument("--budget", nargs="+", default=None, type=float)
    parser.add_argument("--x_range", nargs="+", default=None, type=float)
    parser.add_argument("--log_x", action="store_true")
    parser.add_argument("--log_y", action="store_true")
    parser.add_argument(
        "--filename", type=str, default=None, help="name out pdf file generated"
    )
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument(
        "--ext",
        type=str,
        choices=["pdf", "png"],
        default="pdf",
        help="the file extension or the plot file type",
    )
    parser.add_argument(
        "--plot_default",
        default=False,
        action="store_true",
        help="plots a horizontal line for the prior score if available",
    )
    parser.add_argument(
        "--plot_optimum",
        default=False,
        action="store_true",
        help="plots a horizontal line for the optimum score if available",
    )
    parser.add_argument(
        "--plot_rs_10",
        default=False,
        action="store_true",
        help="plots a horizontal line for RS at 10x",
    )
    parser.add_argument(
        "--plot_rs_25",
        default=False,
        action="store_true",
        help="plots a horizontal line for RS at 25x",
    )
    parser.add_argument(
        "--plot_rs_100",
        default=False,
        action="store_true",
        help="plots a horizontal line for RS at 100x",
    )
    parser.add_argument(
        "--dynamic_y_lim",
        default=False,
        action="store_true",
        help="whether to set y_lim for plots to the worst performance of incumbents"
        "of random_search and random_search_prior after 2 evals"
        "(remember to run it first!)",
    )
    parser.add_argument(
        "--parallel",
        default=False,
        action="store_true",
        help="whether to process data in parallel or not",
    )
    parser.add_argument(
        "--plot_max_fidelity_loss",
        default=False,
        action="store_true",
        help="If set (to True), the incumbent trace is modified such that the loss of "
        "the current incumbent at the max fidelity is plotted instead of the actual "
        "score of the current incumbent.",
    )

    return parser


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


def save_fig(fig, filename, output_dir, extension="pdf", dpi: int = 100):
    output_dir = Path(output_dir)
    output_dir.makedirs_p()
    fig.savefig(output_dir / f"{filename}.{extension}", bbox_inches="tight", dpi=dpi)
    print(f'Saved to "{output_dir}/{filename}.{extension}"')


def interpolate_time(
    incumbents,
    costs,
    x_range=None,
    scale_x=None,
    parallel_evaluations: bool = False,
    rounded_integer_costs_for_x_range: bool = False,
):
    df_dict = {
        f"seed{i}": pd.Series(seed_incs, index=seed_costs)
        for i, (seed_incs, seed_costs) in enumerate(zip(incumbents, costs))
    }
    if not parallel_evaluations:
        for k, series in df_dict.items():
            series.index = np.cumsum(series.index)

    df = pd.DataFrame.from_dict(df_dict)
    df = df.sort_index(ascending=True)

    # important step to plot func evals on x-axis
    df.index = df.index if scale_x is None else df.index.values / scale_x

    if x_range is not None:
        min_b, max_b = x_range

        new_entry = {c: np.nan for c in df.columns}
        _df = pd.DataFrame.from_dict(new_entry, orient="index").T
        _df.index = [min_b]
        df = pd.concat((df, _df)).sort_index()

        new_entry = {c: np.nan for c in df.columns}
        _df = pd.DataFrame.from_dict(new_entry, orient="index").T
        _df.index = [max_b]
        df = pd.concat((df, _df)).sort_index()

    df = df.fillna(method="backfill", axis=0).fillna(method="ffill", axis=0)

    if x_range is not None:
        lower, upper = x_range

        if rounded_integer_costs_for_x_range:
            _index = df.index.astype(int)
        else:
            _index = df.index

        df = df[(lower <= df.index) & (df.index <= upper)]

    return df


def plot_incumbent(
    ax,
    df,
    # x,
    # y,
    xlabel=None,
    ylabel=None,
    title=None,
    algorithm=None,
    log_x=False,
    log_y=False,
    x_range=None,
    # max_cost=None,
    plot_default=None,
    plot_optimum=None,
    plot_rs_10=None,
    plot_rs_25=None,
    plot_rs_100=None,
    force_prior_line=False,
    **plot_kwargs,
):
    # if isinstance(x, list):
    #     x = np.array(x)
    # if isinstance(y, list):
    #     y = np.array(y)

    # df = interpolate_time(incumbents=y, costs=x, x_range=x_range, scale_x=max_cost)

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
        # ax.hlines(y=plot_default, xmin=x[0], xmax=x[-1], color="black")

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
        # ax.hlines(y=plot_optimum, xmin=x[0], xmax=x[-1], color="black", linestyle=":")

    if plot_rs_10 is not None and plot_rs_10 < y_mean[0]:
        # plot only if the optimum score is better than the first incumbent plotted
        ax.plot(x, [plot_rs_10] * len(x), color="grey", linestyle=":", label="RS@10")
        # ax.hlines(y=plot_rs_10, xmin=x[0], xmax=x[-1], color="grey", linestyle=":")

    if plot_rs_25 is not None and plot_rs_25 < y_mean[0]:
        # plot only if the optimum score is better than the first incumbent plotted
        ax.plot(x, [plot_rs_25] * len(x), color="grey", linestyle="-.", label="RS@25")
        # ax.hlines(y=plot_rs_25, xmin=x[0], xmax=x[-1], color="grey", linestyle="-.")

    if plot_rs_100 is not None and plot_rs_100 < y_mean[0]:
        # plot only if the optimum score is better than the first incumbent plotted
        ax.plot(x, [plot_rs_100] * len(x), color="grey", linestyle="--", label="RS@100")
        # ax.hlines(y=plot_rs_100, xmin=x[0], xmax=x[-1], color="grey", linestyle="--")

    ax.fill_between(
        x,
        y_mean - std_error,
        y_mean + std_error,
        color=COLOR_MARKER_DICT[algorithm],
        alpha=0.1,
        step="post",
    )

    ax.set_xlim(auto=True)
    # ax.set_ylim(auto=True)

    if title is not None:
        ax.set_title(DATASETS[title], fontsize=20, color=BENCHMARK_COLORS[title])
    if xlabel is not None:
        ax.set_xlabel(xlabel, fontsize=18, color=(0, 0, 0, 0.69))
    if ylabel is not None:
        ax.set_ylabel(ylabel, fontsize=18, color=(0, 0, 0, 0.69))
    if log_x:
        ax.set_xscale("log")
    if log_y:
        # ax.set_yscale("log")
        ax.set_yscale("symlog")
    if x_range is not None:
        ax.set_xlim(*x_range)
        if x_range == [1, 12]:
            ax.set_xticks([1, 3, 5, 10, 12], [1, 3, 5, 10, 12])

    # Black with some alpha
    ax.tick_params(axis="both", which="major", labelsize=18, labelcolor=(0, 0, 0, 0.69))
    ax.grid(True, which="both", ls="-", alpha=0.8)
