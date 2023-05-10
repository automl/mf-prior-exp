from __future__ import annotations

import sys
import argparse
from dataclasses import dataclass
from itertools import chain, tee
from pathlib import Path
from warnings import warn
from typing import TypeVar, Callable, Iterator, Iterable

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sns.set_theme(style="whitegrid")

HERE = Path(__file__).parent.absolute().resolve()
BENCHMARK_DIR = HERE.parent / "src" / "mf_prior_experiments" / "configs" / "benchmark"
BASEPATH = HERE.parent
RESULTS_DIR = BASEPATH / "results"
PLOT_DIR = BASEPATH / "plots"

T = TypeVar("T")

def prior_renaming(benchmark: str, prior: str) -> str:
    if benchmark.startswith("mfh"):
        renamings = {
            "good": "Near Optimum",
            "at25": "Good",
            "bad": "Bad",
        }
    else:
        renamings = {
            "medium": "Near optimum",
            "at25": "Good",
            "bad": "Bad",
        }
    return renamings.get(prior, prior)




def partition(
    xs: Iterable[T],
    key: Callable[[T], bool],
) -> tuple[Iterator[T], Iterator[T]]:
    conditioned_xs = ((item, key(item)) for item in xs)
    _l, _r = tee(conditioned_xs)
    return (x for x, pred in _l if pred), (x for x, pred in _r if not pred)


@dataclass
class PlotData:
    benchmark: str
    priors: dict[str, list[float]]

    def __post_init__(self) -> None:
        shortest = min(len(values) for values in self.priors.values())
        if not all(len(values) == shortest for values in self.priors.values()):
            warn(f"Length mismatchs, pruning to shortest={shortest}")
            self.priors = {
                name: values[:shortest]
                for name, values in self.priors.items()
            }

    @property
    def empty(self) -> bool:
        return not any(self.priors)

    def df(self) -> pd.DataFrame:
        values = list(chain.from_iterable(self.priors.values()))
        prior_names = list(chain.from_iterable([prior_renaming(self.benchmark, name)] * len(values) for name, values in self.priors.items()))

        return pd.DataFrame(
            {
                "prior": prior_names,
                "error": values
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
    priors: list[str],
) -> PlotData:
    if benchmark.startswith("mfh"):
        # MFH has it's near optimum prior as good
        priors = [p.replace("medium", "good") for p in priors]

    def get(p, s):
        return load_losses(benchmark, algo=algo, prior=p, seed=s, group=group)  # noqa

    prior_evaluations = {
        name: list(chain.from_iterable(get(name, s) for s in seeds))
        for name in priors
    }

    return PlotData(
        benchmark=benchmark,
        priors=prior_evaluations
    )


def plot(
    ax: plt.axes.Axes,
    data: PlotData,
    normalization: str | None = None,
) -> None:
    df = data.df()
    sns.despine(bottom=True, left=True)

    npriors = df["prior"].nunique()

    ax = sns.violinplot(
        data=df,
        x="prior",
        y="error",
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

    # ax.set_xlabel(xlabel, fontsize=18, color=(0, 0, 0, 0.69))
    # ax.set_ylabel("Benchmark", fontsize=18, color=(0, 0, 0, 0.69))
    # ax.legend(fontsize=15, loc="upper center", bbox_to_anchor=(0.5, -0.15), fancybox=True, ncol=npriors)

    #ax.tick_params(axis="both", which="major", labelsize=15, labelcolor=(0, 0, 0, 0.69))
    x_axis = ax.axes.get_xaxis()
    x_label = x_axis.get_label()
    x_label.set_visible(False)

    ax.set_title(data.benchmark)
    #for tick in ax.xaxis.get_major_ticks()[1::2]:
    #    tick.set_pad(18)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="mf-prior-exp plotting for priors")
    parser.add_argument("--benchmarks", nargs="+", type=str, required=True)
    parser.add_argument("--priors", type=str, nargs="+", required=True)
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
    print(args)

    data = [
        results(
            b,
            algo=args.algo,
            seeds=args.seeds,
            group=args.group,
            priors=args.priors,
        )
        for b in args.benchmarks
    ]
    success, failed = partition(data, lambda d: not d.empty)
    success, failed = list(success), list(failed)

    print(f"SUCCESS - {[d.benchmark for d in success]}")
    print(f"FAILED - {[d.benchmark for d in failed]}")

    if not any(success):
        print("No successes")
        sys.exit()

    normalization = None if args.normalization == "None" else args.normalization

    n_benchmarks = (len(success))
    ncols = 4
    nrows = ((n_benchmarks + 1) // ncols)

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, sharey=False, figsize=(7 * ncols, 5.2 * nrows))
    for data, ax in zip(success, axes.flatten()):
        plot(ax, data, normalization=normalization)

    output_path = PLOT_DIR / args.group / f"{args.filename}.{args.ext}"
    fig.savefig(output_path, bbox_inches="tight", dpi=args.dpi)
    print(f'Saved to "{output_path}"')
