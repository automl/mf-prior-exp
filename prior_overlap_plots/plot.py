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


@dataclass
class PlotData:
    benchmark: str
    good: list[float]
    bad: list[float]

    def __post_init__(self) -> None:
        l_good = len(self.good)
        l_bad = len(self.bad)
        if not l_good == l_bad:

            if l_good > l_bad:
                self.good = self.good[: len(self.bad)]
            else:
                self.bad = self.bad[: len(self.good)]

            warn(
                f"Length mismatch, good={l_good} | bad={l_bad})."
                f"  Pruned to good={len(self.good)} | bad = {len(self.bad)}"
            )

    def df(self) -> pd.DataFrame:
        priors = ["good"] * len(self.good) + ["bad"] * len(self.bad)
        error = self.good + self.bad
        return pd.DataFrame(
            {"benchmark": self.benchmark, "prior": priors, "error": error}
        )

    def normalized_df(self) -> pd.DataFrame:
        df = self.df()
        mi = df["error"].min()
        ma = df["error"].max()
        df["error"] = (df["error"] - mi) / (ma - mi)
        return df


def load_losses(
    benchmark: str,
    prior: str,
    algo: str,
    seed: int,
    group: str,
) -> list[float]:
    if benchmark.startswith("mfh"):
        prior = {"good": "perfect-noisy0.125", "bad": "bad-noisy0.125"}[prior]

    path = (
        RESULTS_DIR
        / group
        / f"benchmark={benchmark}_prior-{prior}"
        / f"algorithm={algo}"
        / f"seed={seed}"
        / "neps_root_directory"
        / "all_losses_and_configs.txt"
    )

    with path.open(mode="r", encoding="UTF-8") as f:
        data = f.readlines()

    return [float(e.strip().split("Loss: ")[1]) for e in data if "Loss: " in e]


def results(benchmark: str, algo: str, seeds: list[int], group: str) -> PlotData:
    get = lambda p, s: load_losses(benchmark, algo=algo, prior=p, seed=s, group=group)  # noqa

    good = chain.from_iterable(get("good", s) for s in seeds)
    bad = chain.from_iterable(get("bad", s) for s in seeds)

    return PlotData(benchmark=benchmark, good=list(good), bad=list(bad))


def plot(ax: plt.axes.Axes, data: list[PlotData]) -> None:
    df = pd.concat([d.normalized_df() for d in data])
    sns.violinplot(
        data=df,
        x="benchmark",
        y="error",
        hue="prior",
        ax=ax,
        split=True,
        cut=0,
        inner="stick",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="mf-prior-exp plotting for priors")
    parser.add_argument("--benchmarks", nargs="+", type=str)
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--group", type=str)
    parser.add_argument("--algo", type=str)
    parser.add_argument("--filename", type=str, default="prior_plot")
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument("--ext", type=str, choices=["pdf", "png"], default="pdf")
    args = parser.parse_args()
    print(args)

    data = [results(b, algo=args.algo, seeds=args.seeds, group=args.group) for b in args.benchmarks]

    fig, ax = plt.subplots()
    plot(ax, data)

    output_path = PLOT_DIR / f"{args.filename}.{args.ext}"
    fig.savefig(output_path, bbox_inches="tight", dpi=args.dpi)
    print(f'Saved to "{output_path}"')
