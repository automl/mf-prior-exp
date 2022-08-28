import numpy as np
import pandas as pd
import seaborn as sns
from path import Path
from scipy import stats
import matplotlib.pyplot as plt

from .styles import ALGORITHMS, COLOR_MARKER_DICT, DATASETS


def set_general_plot_style():
    sns.set_style("ticks")
    sns.set_context("paper")
    sns.set_palette("deep")
    plt.switch_backend("pgf")
    plt.rcParams.update(
        {
            "text.usetex": True,
            "pgf.texsystem": "pdflatex",
            "pgf.rcfonts": False,
            "font.family": "serif",
            "font.serif": [],
            "font.sans-serif": [],
            "font.monospace": [],
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
            "pgf.preamble": r"""
                \usepackage[T1]{fontenc}
                \usepackage[utf8x]{inputenc}
                \usepackage{microtype}
            """,
        }
    )


def save_fig(fig, filename, output_dir, dpi: int = 100):
    output_dir = Path(output_dir)
    output_dir.makedirs_p()
    fig.savefig(output_dir / f"{filename}.pdf", bbox_inches="tight", dpi=dpi)
    print(f'Saved to "{output_dir}/{filename}.pdf"')


def interpolate_time(incumbents, costs):
    df_dict = {}

    for i, _ in enumerate(incumbents):
        _seed_info = pd.Series(incumbents[i], index=np.cumsum(costs[i]))
        df_dict[f"seed{i}"] = _seed_info
    df = pd.DataFrame.from_dict(df_dict)

    df = df.fillna(method="backfill", axis=0).fillna(method="ffill", axis=0)
    return df


def plot_incumbent(
    ax,
    x,
    y,
    xlabel=None,
    ylabel=None,
    title=None,
    algorithm=None,
    **plot_kwargs,
):
    if isinstance(x, list):
        x = np.array(x)
    if isinstance(y, list):
        y = np.array(y)

    df = interpolate_time(incumbents=y, costs=x)
    # df = df.iloc[np.linspace(0, len(df) - 1, 1001)]
    x = df.index
    y_mean = df.mean(axis=1)
    std_error = stats.sem(df.values, axis=1)

    ax.plot(
        x,
        y_mean,
        label=ALGORITHMS[algorithm],
        **plot_kwargs,
        color=COLOR_MARKER_DICT[algorithm],
        linewidth=0.7
    )

    ax.fill_between(
        x,
        y_mean - std_error,
        y_mean + std_error,
        color=COLOR_MARKER_DICT[algorithm],
        alpha=0.2,
    )

    if title is not None:
        ax.set_title(DATASETS[title])
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    ax.grid(True, which="both", ls="-", alpha=0.8)
