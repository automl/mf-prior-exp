from __future__ import annotations

import math
import sys
import time
from pathlib import Path
from argparse import ArgumentParser, Namespace

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from typing_extensions import Literal

from .styles import (
    ALGORITHMS,
    BENCHMARK_COLORS,
    COLOR_MARKER_DICT,
    DATASETS,
    X_LABEL,
    Y_LABEL,
    Y_LIMITS,
    get_xticks,
    RC_PARAMS,
)
from .types import ExperimentResults, fetch_results

HERE = Path(__file__).parent.absolute()
DEFAULT_BASE_PATH = HERE.parent.parent

is_last_row = lambda idx, nrows, ncols: idx >= (nrows - 1) * ncols
HERE = Path(__file__).parent.absolute()
is_first_column = lambda idx, ncols: idx % ncols == 0


def now() -> str:
    return time.strftime("%H:%M:%S", time.localtime())


def reorganize_legend(
    fig: plt.Figure,
    axs: list[plt.Axes],
    to_front: list[str],
    bbox_to_anchor: tuple[float, float],
    ncol: int,
) -> None:
    handles, labels = axs[0].get_legend_handles_labels()
    handles_to_plot, labels_to_plot = [], []  # type: ignore
    handles_default, labels_default = [], []  # type: ignore
    for h, l in zip(handles, labels):
        if l not in (labels_to_plot + labels_default):
            if l.lower() in to_front:
                handles_default.append(h)
                labels_default.append(l)
            else:
                handles_to_plot.append(h)
                labels_to_plot.append(l)

    handles_to_plot = handles_default + handles_to_plot
    labels_to_plot = labels_default + labels_to_plot

    leg = fig.legend(
        handles_to_plot,
        labels_to_plot,
        fontsize="xx-large",
        loc="lower center",
        bbox_to_anchor=bbox_to_anchor,
        ncol=ncol,
        frameon=True,
    )

    for legend_item in leg.legendHandles:
        legend_item.set_linewidth(2.0)


def plot_relative_ranks(
    algorithms: list[str],
    yaxis: str,
    xaxis: str,
    good_corr_good_prior: ExperimentResults | None = None,
    good_corr_bad_prior: ExperimentResults | None = None,
    bad_corr_good_prior: ExperimentResults | None = None,
    bad_corr_bad_prior: ExperimentResults | None = None,
    pairwise_plots: tuple[tuple[str, ExperimentResults], tuple[str, ExperimentResults]]
    | None = None,
    x_together: float | None = None,
    x_range: tuple[int, int] | None = None,
) -> plt.Figure:
    """Plot relative ranks of the incumbent over time."""

    if pairwise_plots:
        assert good_corr_good_prior is None
        assert good_corr_bad_prior is None
        assert bad_corr_good_prior is None
        assert bad_corr_bad_prior is None
        subplots = {title: results for title, results in pairwise_plots}
    else:
        assert good_corr_good_prior is not None
        assert good_corr_bad_prior is not None
        assert bad_corr_good_prior is not None
        assert bad_corr_bad_prior is not None
        subplots = {
            "good corr. & good prior": good_corr_good_prior,
            "good corr. & bad prior": good_corr_bad_prior,
            "bad corr. & good prior": bad_corr_good_prior,
            "bad corr. & bad prior": bad_corr_bad_prior,
        }

    for key, value in subplots.items():
        print(key, value.results.keys())

    ncols = len(subplots)
    nrows = 1
    figsize = (ncols * 4, nrows * 3)
    legend_ncol = len(algorithms)

    fig, _axs = plt.subplots(nrows, ncols, figsize=figsize)
    axs: list[plt.Axes] = list(_axs.flatten())

    for col, ((subtitle, results), ax) in enumerate(zip(subplots.items(), axs)):
        _x_range: tuple[int, int]
        if x_range is None:
            xmin = min(getattr(r, xaxis) for r in results.iter_results())
            xmax = max(getattr(r, xaxis) for r in results.iter_results())
            _x_range = (math.floor(xmin), math.ceil(xmax))
        else:
            _x_range = tuple(x_range)  # type: ignore

        left, right = _x_range
        xticks = get_xticks(_x_range)
        ymin, ymax = (0.8, len(algorithms))
        yticks = range(1, len(algorithms) + 1)
        center = (len(algorithms) + 1) / 2

        ax.set_title(subtitle)
        ax.set_ylim(ymin, ymax)
        ax.set_xlabel(X_LABEL[xaxis], fontsize=18, color=(0, 0, 0, 0.69))
        ax.set_yticks(yticks)  # type: ignore
        ax.set_xlim(left=left, right=right)
        ax.set_xticks(xticks, xticks)  # type: ignore
        ax.tick_params(axis="both", which="major", labelsize=18, color=(0, 0, 0, 0.69))
        ax.grid(True, which="both", ls="-", alpha=0.8)

        if col == 0:
            ax.set_ylabel("Relative rank", fontsize=18, color=(0, 0, 0, 0.69))

        all_means, all_stds = results.ranks(xaxis=xaxis, yaxis=yaxis)

        for algorithm in algorithms:
            means: pd.Series = all_means[algorithm]  # type: ignore
            stds: pd.Series = all_stds[algorithm]  # type: ignore

            # If x_together is specified, we want to shave off
            # everything in the x-axis before the x_together index
            # so that it lines up with the above
            if x_together is not None:
                means = means.loc[x_together:]
                stds = stds.loc[x_together:]
            elif x_together is None:
                # Otherwise, we just use whatever the xaxis cutoff is
                means = means.loc[left:]
                stds = stds.loc[left:]

            # Center everything
            means.loc[0] = center
            stds.loc[0] = 0

            means = means.sort_index(ascending=True)  # type: ignore
            stds = stds.sort_index(ascending=True)  # type: ignore
            assert means is not None
            assert stds is not None

            x = np.array(means.index.tolist(), dtype=float)
            y = np.array(means.tolist(), dtype=float)
            std = np.array(stds.tolist(), dtype=float)

            ax.step(
                x=x,
                y=y,
                color=COLOR_MARKER_DICT[algorithm],
                linewidth=1,
                where="post",
                label=ALGORITHMS[algorithm],
            )
            ax.fill_between(
                x,
                y - std,  # type: ignore
                y + std,  # type: ignore
                color=COLOR_MARKER_DICT[algorithm],
                alpha=0.1,
                step="post",
            )

    sns.despine(fig)
    handles, labels = axs[0].get_legend_handles_labels()
    legend = fig.legend(
        handles,
        labels,
        fontsize="xx-large",
        loc="lower center",
        bbox_to_anchor=(0.5, -0.20),
        ncol=legend_ncol,
        frameon=True,
    )

    for item in legend.legendHandles:
        item.set_linewidth(2)

    fig.tight_layout(pad=0, h_pad=0.5)
    return fig


def plot_incumbent_traces(
    results: ExperimentResults,
    plot_default: bool = True,
    plot_optimum: bool = True,
    yaxis: Literal["loss", "max_fidelity_loss"] = "loss",
    xaxis: Literal[
        "cumulated_fidelity",
        "end_time_since_global_start",
    ] = "cumulated_fidelity",
    x_range: tuple[int, int] | None = None,
    dynamic_y_lim: bool = False,
) -> plt.Figure:
    benchmarks = results.benchmarks
    algorithms = results.algorithms
    bench_configs = results.benchmark_configs
    all_indices = pd.Index(results.indices(xaxis=xaxis, sort=False))

    # We only enable the option if the benchmark has these recorded
    plot_default = plot_default and any(
        c.prior_error is not None for c in bench_configs.values()
    )

    plot_optimum = plot_optimum and any(
        c.optimum is not None for c in bench_configs.values()
    )

    nrows = np.ceil(len(benchmarks) / 4).astype(int)
    ncols = min(len(benchmarks), 4)
    legend_ncol = len(algorithms) + sum([plot_default, plot_optimum])
    figsize = (4 * ncols, 3 * nrows)

    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
    axs = list(axs.flatten()) if isinstance(axs, np.ndarray) else [axs]

    for i, benchmark in enumerate(benchmarks):
        benchmark_config = results.benchmark_configs[benchmark]
        benchmark_results = results[benchmark]

        ax = axs[i]
        xlabel = X_LABEL[xaxis] if is_last_row(i, nrows, ncols) else None
        ylabel = Y_LABEL if is_first_column(i, ncols) else None

        _x_range: tuple[int, int]
        if x_range is None:
            xmin = min(getattr(r, xaxis) for r in benchmark_results.iter_results())
            xmax = max(getattr(r, xaxis) for r in benchmark_results.iter_results())
            _x_range = (math.floor(xmin), math.ceil(xmax))
        else:
            _x_range = x_range

        left, right = _x_range

        # Now that we've plotted all algorithms for the benchmark,
        # we need to set some dynamic limits
        if dynamic_y_lim:
            y_values = [
                getattr(result, yaxis)
                for result in benchmark_results.iter_results()
                if getattr(results, xaxis) >= left and getattr(results, xaxis) <= right
            ]
            y_min, y_max = min(y_values), max(y_values)
            dy = abs(y_max - y_min)

            plot_offset = 0.15
            ax.set_ylim(y_min - dy * plot_offset, y_max + dy * plot_offset)
        else:
            ylims = Y_LIMITS.get(benchmark)
            if ylims is None:
                ax.set_ylim(auto=True)
            else:
                down, up = ylims
                ax.set_ylim(down, up)

        ax.set_xlim(left=left, right=right)
        if (left, right) == (1, 12):
            xticks = [1, 3, 5, 8, 12]
        else:
            xticks = np.linspace(left, right, 5, dtype=int, endpoint=True).tolist()
        ax.set_xticks(xticks, xticks)

        ax.set_title(
            DATASETS.get(benchmark, benchmark),
            fontsize=20,
            color=BENCHMARK_COLORS.get(benchmark, "black"),
        )

        ax.set_xlabel(xlabel, fontsize=18, color=(0, 0, 0, 0.69))
        ax.set_ylabel(ylabel, fontsize=18, color=(0, 0, 0, 0.69))

        # Black with some alpha
        ax.tick_params(
            axis="both", which="major", labelsize=18, labelcolor=(0, 0, 0, 0.69)
        )
        ax.grid(True, which="both", ls="-", alpha=0.8)

        if plot_default and benchmark_config.prior_error is not None:
            # NOTE: In the case of MFH good where we have taken a prior close
            # to the optimum, and additionally add 0.25 noise at every iteration,
            # there is no predefined prior line we can meaningfully plot. Each
            # run will see a different prior. For consistency in the plots, we
            # have chosen to take the mean line of RS+Prior as a proxy to the
            # averaged prior, as RS+Prior will always sample the prior first.
            mfh_good_prior_benchmarks = [
                "mfh3_good_prior-good",
                "mfh3_terrible_prior-good",
                "mfh6_good_prior-good",
                "mfh6_terrible_prior-good"
            ]
            if "random_search_prior" in algorithms and benchmark in mfh_good_prior_benchmarks:
                random_search_results = benchmark_results["random_search_prior"]
                values = random_search_results.df(index=xaxis, values=yaxis)
                prior_error = values.iloc[0].mean(axis=0)
            else:
                prior_error = benchmark_config.prior_error

            ax.axhline(
                prior_error,
                color="black",
                linestyle=":",
                linewidth=1.0,
                dashes=(5, 10),
                label="Mode",
            )

        if plot_optimum and benchmark_config.optimum is not None:
            # plot only if the optimum score is better than the first incumbent plotted
            ax.axhline(
                benchmark_config.optimum,
                color="black",
                linestyle="-.",
                linewidth=1.2,
                label="Optimum",
            )

        for algorithm in algorithms:
            print("-" * 50)
            print(f"Benchmark: {benchmark} | Algorithm: {algorithm}")
            print("-" * 50)
            df = benchmark_results[algorithm].df(index=xaxis, values=yaxis)
            assert isinstance(df, pd.DataFrame)

            missing_indices = all_indices.difference(df.index)
            if missing_indices is not None:
                for missing_i in missing_indices:
                    df.loc[missing_i] = np.nan

            df = df.sort_index(ascending=True)
            assert df is not None

            df = df.fillna(method="ffill", axis=0)

            x = df.index
            y_mean = df.mean(axis=1).values
            std_error = stats.sem(df.values, axis=1)

            ax.step(
                x,
                y_mean,
                label=ALGORITHMS[algorithm],
                color=COLOR_MARKER_DICT[algorithm],
                linestyle="-",
                linewidth=1,
                where="post",
            )
            ax.fill_between(
                x,
                y_mean - std_error,
                y_mean + std_error,
                color=COLOR_MARKER_DICT.get(algorithm, "black"),
                alpha=0.1,
                step="post",
            )

    bbox_y_mapping = {1: -0.20, 2: -0.11, 3: -0.07, 4: -0.05, 5: -0.04}
    reorganize_legend(
        fig=fig,
        axs=axs,
        to_front=["Mode", "Optimum"],
        bbox_to_anchor=(0.5, bbox_y_mapping[nrows]),
        ncol=legend_ncol,
    )

    sns.despine(fig)
    fig.tight_layout(pad=0, h_pad=0.5)

    return fig


def main():
    args = parse_args()

    experiment_group: str = args.experiment_group
    algorithms: list[str] = args.algorithms
    base_path: Path = args.base_path or DEFAULT_BASE_PATH

    benchmarks: set[str]
    benchmarks = set(
        [
            *args.rr_good_corr_good_prior,
            *args.rr_good_corr_bad_prior,
            *args.rr_bad_corr_good_prior,
            *args.rr_bad_corr_bad_prior,
            *args.benchmarks,
        ]
    )

    plot_dir = base_path / "plots" / experiment_group

    xaxis = "cumulated_fidelity"
    yaxes = ["loss", "max_fidelity_loss"]

    # Fetch the results we need
    starttime = time.time()
    print(f"[{now()}] Processing ...")
    results = fetch_results(
        experiment_group=experiment_group,
        benchmarks=list(benchmarks),
        algorithms=algorithms,
        base_path=base_path,  # Base path of the repo
        parallel=args.parallel,  # Whether to process in parallel
        n_workers=args.n_workers,  # Flag to indicate if it was a parallel setup
        continuations=True,  # Continue on fidelities from configurations
        cumulate_fidelities=True,  # Accumulate fidelities in the indices
        xaxis=xaxis,  # The x-axis to use
        rescale_xaxis="max_fidelity",  # We always rescale the xaxis by max_fidelity
        incumbent_value="loss",  # The incumbent is deteremined by the loss
        incumbents_only=True,  # We only want incumbent traces in our results
        use_cache=args.use_cache,  # Specify we want to use cached results
        collect=args.collect,  # Specify we only want to collect
    )
    print(f"[{now()}] Done! Duration {time.time() - starttime:.3f}...")

    # If we are just collecting results, then exit out now
    if args.collect:
        print("Collected results and cached!")
        sys.exit(0)

    # Incumbent traces
    if len(args.benchmarks) > 0:
        for yaxis in yaxes:
            fig = plot_incumbent_traces(
                results=results.select(benchmarks=args.benchmarks),
                plot_default=args.plot_default,
                plot_optimum=args.plot_optimum,
                yaxis=yaxis,  # type: ignore
                xaxis=xaxis,  # type: ignore
                x_range=args.x_range,
            )

            filename = f"{args.filename}.{args.ext}"
            filepath = plot_dir / "incumbent_traces" / yaxis / filename
            filepath.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(filepath, bbox_inches="tight", dpi=args.dpi)
            print(f"Saved to {filename} to {filepath}")

    # Relative ranking plots
    # If one is set, the rest is set
    if len(args.rr_good_corr_good_prior) > 0:
        for yaxis in yaxes:
            fig = plot_relative_ranks(
                algorithms=algorithms,
                good_corr_good_prior=results.select(
                    benchmarks=args.rr_good_corr_good_prior
                ),
                good_corr_bad_prior=results.select(
                    benchmarks=args.rr_good_corr_bad_prior
                ),
                bad_corr_good_prior=results.select(
                    benchmarks=args.rr_bad_corr_good_prior
                ),
                bad_corr_bad_prior=results.select(benchmarks=args.rr_bad_corr_bad_prior),
                yaxis=yaxis,
                xaxis=xaxis,
                x_range=args.x_range,
                x_together=args.x_together,
            )

            filename = f"{args.filename}.{args.ext}"
            filepath = plot_dir / "relative_ranks" / yaxis / filename
            filepath.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(filepath, bbox_inches="tight", dpi=args.dpi)
            print(f"Saved to {filename} to {filepath}")

        # We also do a relative ranking plot in two sections
        # aggrgating by prior kind
        good_prior_benchmarks = [
            *args.rr_good_corr_good_prior,
            *args.rr_bad_corr_good_prior,
        ]
        bad_prior_benchmarks = [*args.rr_good_corr_bad_prior, *args.rr_bad_corr_bad_prior]
        print(f"good_prior_benchmarks={good_prior_benchmarks}")
        print(f"bad_prior_benchmarks={bad_prior_benchmarks}")

        for yaxis in yaxes:
            fig = plot_relative_ranks(
                algorithms=algorithms,
                pairwise_plots=(
                    ("good prior", results.select(benchmarks=good_prior_benchmarks)),
                    ("bad prior", results.select(benchmarks=bad_prior_benchmarks)),
                ),
                yaxis=yaxis,
                xaxis=xaxis,
                x_range=args.x_range,
                x_together=args.x_together,
            )
            filename = f"{args.filename}-benchmarks-by-prior.{args.ext}"
            filepath = plot_dir / "relative_ranks" / yaxis / filename
            filepath.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(filepath, bbox_inches="tight", dpi=args.dpi)
            print(f"Saved to {filename} to {filepath}")

        # We also do a relative ranking plot in two sections
        # aggrgating by correlation kind
        good_corr_benchmarks = [
            *args.rr_good_corr_good_prior,
            *args.rr_good_corr_bad_prior,
        ]
        bad_corr_benchmarks = [*args.rr_bad_corr_good_prior, *args.rr_bad_corr_bad_prior]
        print(f"good_corr_benchmarks={good_corr_benchmarks}")
        print(f"bad_corr_benchmarks={bad_corr_benchmarks}")

        for yaxis in yaxes:
            fig = plot_relative_ranks(
                algorithms=algorithms,
                pairwise_plots=(
                    ("good corr.", results.select(benchmarks=good_corr_benchmarks)),
                    ("bad corr.", results.select(benchmarks=bad_corr_benchmarks)),
                ),
                yaxis=yaxis,
                xaxis=xaxis,
                x_range=args.x_range,
                x_together=args.x_together,
            )
            filename = f"{args.filename}-benchmarks-by-correlation.{args.ext}"
            filepath = plot_dir / "relative_ranks" / yaxis / filename
            filepath.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(filepath, bbox_inches="tight", dpi=args.dpi)
            print(f"Saved to {filename} to {filepath}")


def parse_args() -> Namespace:
    plt.rcParams.update(RC_PARAMS)
    parser = ArgumentParser(description="mf-prior-exp plotting")

    parser.add_argument("--filename", type=str, default=None)
    parser.add_argument("--collect", action="store_true")
    parser.add_argument("--use-cache", action="store_true")

    parser.add_argument("--experiment_group", type=str, required=True)
    parser.add_argument("--algorithms", nargs="+", default=[])

    parser.add_argument("--benchmarks", nargs="+", default=[])
    parser.add_argument("--rr-good-corr-good-prior", nargs="+", default=[])
    parser.add_argument("--rr-good-corr-bad-prior", nargs="+", default=[])
    parser.add_argument("--rr-bad-corr-good-prior", nargs="+", default=[])
    parser.add_argument("--rr-bad-corr-bad-prior", nargs="+", default=[])

    parser.add_argument("--base_path", type=Path, default=None)
    parser.add_argument("--n_workers", type=int, default=1)
    parser.add_argument("--budget", nargs="+", type=float, default=None)
    parser.add_argument("--x_range", nargs=2, type=float, default=None)
    parser.add_argument("--x_together", type=float, default=None)

    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument("--ext", type=str, choices=["pdf", "png"], default="png")
    parser.add_argument("--plot_default", action="store_true")
    parser.add_argument("--plot_optimum", action="store_true")
    #parser.add_argument("--dynamic_y_lim", action="store_true")
    parser.add_argument("--parallel", action="store_true")

    args = parser.parse_args()

    if args.collect and args.use_cache:
        raise ValueError("Can't use --collect with --use-cache")

    if not args.collect and args.filename is None:
        raise ValueError("Must specify --filename unless using --collect")

    if args.x_together and args.x_range and args.x_together < args.x_range[0]:
        raise ValueError("--x_together must be larger than --x_range[0]")

    if not args.collect and len(args.algorithms) == 0:
        raise ValueError("Must specify --algorithms unless using --collect")

    if args.budget:
        raise ValueError("CD plots (which use --budget) not supported yet")

    benches = [
        args.rr_good_corr_good_prior,
        args.rr_good_corr_bad_prior,
        args.rr_bad_corr_good_prior,
        args.rr_bad_corr_bad_prior,
    ]
    if any(len(b) > 0 for b in benches) and not all(len(b) > 0 for b in benches):
        raise ValueError("Must specify all --rr args for relative rankings")

        #if not len(list(flatten(benches))) == len(set(flatten(benches))):
        #raise ValueError("Benchmarks in --rr must all be unique\n")

    return args

if __name__ == "__main__":
    main()
