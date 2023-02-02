from __future__ import annotations

import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from typing_extensions import Literal

from .configs.plotting.styles import X_LABEL, Y_LABEL
from .configs.plotting.types import ExperimentResults, fetch_results
from .configs.plotting.utils import (
    parse_args,
    plot_incumbent,
    set_general_plot_style,
)

HERE = Path(__file__).parent.absolute()
DEFAULT_BASE_PATH = HERE.parent.parent


def now() -> str:
    return time.strftime("%H:%M:%S", time.localtime())


def plot_incumbent_traces(
    results: ExperimentResults,
    plot_default: bool = True,
    plot_optimum: bool = True,
    yaxis: Literal["loss", "max_fidelity_loss"] = "loss",
    xaxis: Literal[
        "single_worker_cumulated_fidelity",
        "end_time_since_global_start",
    ] = "single_worker_cumulated_fidelity",
    x_range: tuple[float, float] | None = None,
    dynamic_y_lim: bool = True,
) -> plt.Figure:
    benchmarks = results.benchmarks
    algorithms = results.algorithms
    benchmark_configs = results.benchmark_configs

    # If we are going to plot the optimum, we need to make sure
    # there is a benchmark that actually has an optimum to plot
    if plot_optimum and not any(
        bench_config.optimum for bench_config in benchmark_configs.values()
    ):
        plot_optimum = False

    legend_ncol = len(algorithms) + sum([plot_default, plot_optimum])

    nrows = np.ceil(len(benchmarks) / 4).astype(int)
    ncols = min(len(benchmarks), 4)
    is_last_row = lambda idx: idx >= (nrows - 1) * ncols
    is_first_column = lambda idx: idx % ncols == 0

    bbox_y_mapping = {1: -0.20, 2: -0.11, 3: -0.07, 4: -0.05, 5: -0.04}
    bbox_to_anchor = (0.5, bbox_y_mapping[nrows])
    figsize = (4 * ncols, 3 * nrows)

    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
    axs = axs.flatten() if isinstance(axs, np.ndarray) else [axs]

    for i, benchmark in enumerate(benchmarks):
        benchmark_config = results.benchmark_configs[benchmark]
        benchmark_results = results[benchmark]

        ax = axs[i]
        xlabel = X_LABEL[xaxis] if is_last_row(i) else None
        ylabel = Y_LABEL if is_first_column(i) else None


        for algorithm in algorithms:
            print("-" * 50)
            print(f"Benchmark: {benchmark} | Algorithm: {algorithm}")
            print("-" * 50)
            df = (
                benchmark_results[algorithm]
                .df(index=xaxis, values=yaxis)
                .fillna(method="ffill", axis=0)
            )

            plot_incumbent(
                ax=ax,
                df=df,
                title=benchmark,
                xlabel=xlabel,
                ylabel=ylabel,
                algorithm=algorithm,
                x_range=x_range,
                plot_default=benchmark_config.prior_error if plot_default else None,
                plot_optimum=benchmark_config.optimum if plot_optimum else None,
                force_prior_line="good" in benchmark_config.name,
            )

    # Now that we've plotted all algorithms for the benchmark,
    # we need to set some dynamic limits
    if dynamic_y_lim:
        for ax, benchmark in zip(axs, benchmarks):
            benchmark_results = results[benchmark]
            y_values = [getattr(result, yaxis) for result in benchmark_results.values()]
            y_min, y_max = min(y_values), max(y_values)
            dy = abs(y_max - y_min)

            plot_offset = 0.15
            ax.set_ylim(y_min - dy * plot_offset, y_max + dy * plot_offset)
    else:
        for ax in axs:
            ax.set_ylim(auto=True)

    sns.despine(fig)

    # Move `mode` and `optimum` to the front of the legend
    handles, labels = axs[0].get_legend_handles_labels()
    handles_to_plot, labels_to_plot = [], []  # type: ignore
    handles_default, labels_default = [], []  # type: ignore
    for h, l in zip(handles, labels):
        if l not in (labels_to_plot + labels_default):
            if l.lower() in ["mode", "optimum"]:
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
        ncol=legend_ncol,
        frameon=True,
    )

    for legend_item in leg.legendHandles:
        legend_item.set_linewidth(2.0)

    fig.tight_layout(pad=0, h_pad=0.5)
    return fig



if __name__ == "__main__":
    args = parse_args()

    experiment_group: str = args.experiment_group
    algorithms: list[str] = args.algorithms
    base_path: Path = args.base_path or DEFAULT_BASE_PATH

    benchmarks: list[str]
    if args.relative_rankings:
        benchmarks = [
            *args.benchmarks1,
            *args.benchmarks2,
            *args.benchmarks3,
            *args.benchmarks4,
        ]
    else:
        benchmarks = args.benchmarks

    plot_dir = base_path / "plots" / experiment_group
    set_general_plot_style()

    if args.n_workers == 1:
        xaxis = "single_worker_cumulated_fidelity"
    elif args.n_workers > 1:
        xaxis = "end_time_since_global_start"
    else:
        raise ValueError(f"n_workers={args.n_workers}")

    # Fetch the results we need
    starttime = time.time()
    print(f"[{now()}] Processing ...")
    experiment_results = fetch_results(
        experiment_group=experiment_group,
        benchmarks=benchmarks,
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
    )
    print(f"[{now()}] Done! Duration {time.time() - starttime:.3f}...")

    for yaxis in ["loss", "max_fidelity_loss"]:
        fig = plot_incumbent_traces(
            results=experiment_results,
            plot_default=args.plot_default,
            plot_optimum=args.plot_optimum,
            yaxis=yaxis,  # type: ignore
            xaxis=xaxis,  # type: ignore
            x_range=args.x_range,
            dynamic_y_lim=args.dynamic_y_lim,
        )

        filename = f"{args.filename}.{args.ext}"
        filepath = plot_dir / yaxis / filename
        filepath.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(filepath, bbox_inches="tight", dpi=args.dpi)
        print(f"Saved to {filename} to {filepath}")
