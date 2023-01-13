from __future__ import annotations

import argparse
from dataclasses import dataclass
from itertools import chain
from pathlib import Path
from warnings import warn

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

HERE = Path(__file__).parent.absolute().resolve()
BENCHMARK_DIR = HERE.parent / "src" / "mf_prior_experiments" / "configs" / "benchmark"
BASEPATH = HERE.parent
RESULTS_DIR = BASEPATH / "results"
PLOT_DIR = BASEPATH / "plots"
FAILED_TXT_FILE = HERE / "failed.txt"


@dataclass
class PlotData:
    benchmark: str
    # The left side of the violin
    left: tuple[str, list[float]]
    # The right side of the violin
    right: tuple[str, list[float]]

    def __post_init__(self) -> None:
        left_name, left_values = self.left
        right_name, right_values = self.right

        if len(left_values) != len(right_values):
            shortest = min(len(left_values), len(right_values))
            self.left = (left_name, left_values[:shortest])
            self.right = (right_name, right_values[:shortest])

            warn(
                f"Length mismatchs,"
                f" left={len(left_values)} | right={len(right_values)})."
                f" Pruned to shortest={shortest}"
            )

    def df(self) -> pd.DataFrame:
        left_name, left_values = self.left
        right_name, right_values = self.right
        l_left = len(left_values)
        l_right = len(right_values)

        return pd.DataFrame(
            {
                "benchmark": self.benchmark,
                "prior": [left_name] * l_left + [right_name] * l_right,
                "error": left_values + right_values,
            }
        )

    def min_max_normalized_df(self) -> pd.DataFrame:
        df = self.df()
        mi = df["error"].min()
        ma = df["error"].max()
        df["error"] = (df["error"] - mi) / (ma - mi)
        return df

    def standard_normalized_df(self) -> pd.DataFrame:
        from scipy import stats

        df = self.df()
        df["error"] = stats.zscore(df["error"])
        return df


def load_losses(
    benchmark: str,
    prior: str,
    algo: str,
    seed: int,
    group: str,
) -> list[float]:
    path = (
        RESULTS_DIR
        / group
        / f"benchmark={benchmark}_prior-{prior}"
        / f"algorithm={algo}"
        / f"seed={seed}"
        / "neps_root_directory"
        / "all_losses_and_configs.txt"
    )
    if not path.exists():
        print(f"{benchmark}_prior-{prior} {seed}")
        return []

    with path.open(mode="r", encoding="UTF-8") as f:
        data = f.readlines()

    return [float(e.strip().split("Loss: ")[1]) for e in data if "Loss: " in e]


def results(
    benchmark: str,
    algo: str,
    seeds: list[int],
    group: str,
    left_prior: str,
    right_prior: str,
) -> PlotData:
    def get(p, s):
        return load_losses(benchmark, algo=algo, prior=p, seed=s, group=group)  # noqa

    left_values = list(chain.from_iterable(get(left_prior, s) for s in seeds))
    right_values = list(chain.from_iterable(get(right_prior, s) for s in seeds))

    return PlotData(
        benchmark=benchmark,
        left=(left_prior, left_values),
        right=(right_prior, right_values),
    )


def plot(
    ax: plt.axes.Axes, data: list[PlotData], normalization: str | None = None
) -> None:
    if normalization == "standard":
        df = pd.concat([d.standard_normalized_df() for d in data])
        ylabel = "Error (zscore normalized)"
    elif normalization == "min-max":
        df = pd.concat([d.min_max_normalized_df() for d in data])
        ylabel = "Error (min-max normalized)"
    else:
        df = pd.concat([d.df() for d in data])
        ylabel = "Error"

    ax = sns.violinplot(
        data=df,
        x="benchmark",
        y="error",
        hue="prior",
        ax=ax,
        split=True,
        cut=0,
        inner="quartile",
    )
    # https://stackoverflow.com/q/60638344
    for line in ax.lines:
        line.set_linestyle("--")
        line.set_linewidth(1.2)
        line.set_color("white")

    for line in ax.lines[1::3]:
        line.set_linestyle("-")
        line.set_linewidth(1.8)
        line.set_color("black")

    ax.set_ylabel(ylabel, fontsize=18, color=(0, 0, 0, 0.69))
    # ax.set_xlabel("Benchmark", fontsize=18, color=(0, 0, 0, 0.69))
    ax.xaxis.label.set_visible(False)
    ax.legend(fontsize=15)

    ax.tick_params(axis="both", which="major", labelsize=15, labelcolor=(0, 0, 0, 0.69))
    for tick in ax.xaxis.get_major_ticks()[1::2]:
        tick.set_pad(18)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="mf-prior-exp plotting for priors")
    parser.add_argument("--benchmarks", nargs="+", type=str, required=True)
    parser.add_argument("--left-prior", type=str, required=True)
    parser.add_argument("--right-prior", type=str, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--group", type=str, required=True)
    parser.add_argument("--algo", type=str, required=True)
    parser.add_argument("--filename", type=str, default="prior_plot")
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument("--ext", type=str, choices=["pdf", "png"], default="pdf")
    parser.add_argument(
        "--normalization",
        type=str,
        choices=["standard", "min-max", "None"],
        default="None",
    )
    args = parser.parse_args()

    data = [
        results(
            b,
            algo=args.algo,
            seeds=args.seeds,
            group=args.group,
            left_prior=args.left_prior,
            right_prior=args.right_prior,
        )
        for b in args.benchmarks
    ]
    normalization = None if args.normalization == "None" else args.normalization

    fig, ax = plt.subplots()
    plot(ax, data, normalization=normalization)

    output_path = PLOT_DIR / args.group / f"{args.filename}.{args.ext}"
    fig.savefig(output_path, bbox_inches="tight", dpi=args.dpi)
    print(f'Saved to "{output_path}"')
