import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from path import Path
from scipy import stats

from .styles import ALGORITHMS, COLOR_MARKER_DICT, DATASETS


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


def interpolate_time(incumbents, costs, x_range=None, scale_x=None):
    df_dict = {}

    for i, _ in enumerate(incumbents):
        _seed_info = pd.Series(incumbents[i], index=np.cumsum(costs[i]))
        df_dict[f"seed{i}"] = _seed_info
    df = pd.DataFrame.from_dict(df_dict)

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
        df = df.query(f"{x_range[0]} <= index <= {x_range[1]}")

    return df


def plot_incumbent(
    ax,
    x,
    y,
    xlabel=None,
    ylabel=None,
    title=None,
    algorithm=None,
    log_x=False,
    log_y=False,
    x_range=None,
    max_cost=None,
    plot_default=None,
    **plot_kwargs,
):
    if isinstance(x, list):
        x = np.array(x)
    if isinstance(y, list):
        y = np.array(y)

    df = interpolate_time(incumbents=y, costs=x, x_range=x_range, scale_x=max_cost)

    x = df.index
    y_mean = df.mean(axis=1).values
    std_error = stats.sem(df.values, axis=1)

    ax.plot(
        x,
        y_mean,
        label=ALGORITHMS[algorithm],
        **plot_kwargs,
        # color=COLOR_MARKER_DICT[algorithm],
        linewidth=0.7,
    )
    if plot_default is not None and plot_default < y_mean[0]:
        # plot only if the default score is better than the first incumbent plotted
        ax.hlines(y=plot_default, xmin=x[0], xmax=x[-1], color="black")
    ax.fill_between(
        x,
        y_mean - std_error,
        y_mean + std_error,
        # color=COLOR_MARKER_DICT[algorithm],
        alpha=0.2,
    )

    ax.set_xlim(auto=True)
    ax.set_ylim(auto=True)

    if title is not None:
        ax.set_title(DATASETS[title])
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    if log_x:
        ax.set_xscale("log")
    if log_y:
        # ax.set_yscale("log")
        ax.set_yscale("symlog")
    ax.grid(True, which="both", ls="-", alpha=0.8)
