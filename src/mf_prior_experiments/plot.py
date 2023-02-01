from __future__ import annotations

import time
from contextlib import nullcontext
from multiprocessing import Pool
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from attrdict import AttrDict

from .configs.plotting.styles import X_LABEL, Y_LABEL
from .configs.plotting.types import ExperimentResults
from .configs.plotting.utils import (
    get_parser,
    plot_incumbent,
    save_fig,
    set_general_plot_style,
)

HERE = Path(__file__).parent.absolute()
DEFAULT_BASE_PATH = HERE.parent.parent
BENCHMARK_CONFIGS_DIR = HERE / "configs" / "benchmark"


def now() -> str:
    return time.strftime("%H:%M:%S", time.localtime())


def plot(args):
    BASE_PATH = DEFAULT_BASE_PATH if args.base_path is None else args.base_path
    BENCHMARK_CONFIG_DIR = BASE_PATH / "src" / "mf_prior_experiments" / "configs" / "benchmark"
    RESULTS_DIR = BASE_PATH / "results" / args.experiment_group
    plot_dir = BASE_PATH / "plots" / args.experiment_group

    set_general_plot_style()
    xrange = args.x_range

    if args.research_question == 1:
        ncols = 1 if len(args.benchmarks) == 1 else 2
        legend_ncol = len(args.algorithms)
        legend_ncol += 1 if args.plot_default is not None else 0
        legend_ncol += 1 if args.plot_optimum is not None else 0
    elif args.research_question == 2:
        ncols = 4
        legend_ncol = len(args.algorithms)
        legend_ncol += 1 if args.plot_default is not None else 0
        legend_ncol += 1 if args.plot_optimum is not None else 0
    else:
        raise ValueError("Plotting works only for RQ1 and RQ2.")

    nrows = np.ceil(len(args.benchmarks) / ncols).astype(int)
    bbox_y_mapping = {1: -0.20, 2: -0.11, 3: -0.07, 4: -0.05, 5: -0.04}
    bbox_to_anchor = (0.5, bbox_y_mapping[nrows])
    figsize = (4 * ncols, 3 * nrows)

    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
    axs = axs.flatten() if isinstance(axs, np.ndarray) else [axs]

    # TODO: We may want to change this with the new "continuation fidelity"
    # for parallel setup
    xaxis = "fidelity" if args.n_workers <= 1 else "end_time_since_global_start"
    yaxis = "max_fidelity_loss" if args.plot_max_fidelity_loss else "loss"

    context = Pool() if args.parallel else nullcontext(None)  # noqa: consider-using-with

    starttime = time.time()
    print(f"[{now()}] Processing ...")
    with context as pool:
        experiment_results = ExperimentResults.load(
            path=RESULTS_DIR,
            benchmarks=args.benchmarks,
            algorithms=args.algorithms,
            benchmark_config_dir=BENCHMARK_CONFIG_DIR,
            pool=pool,
        )

        experiment_results = experiment_results.with_continuations(pool=pool)

        if args.n_workers <= 1:
            # fidelities: [1, 1, 3, 1, 9] -> [1, 2, 5, 6, 15]
            experiment_results = experiment_results.with_cumulative_fidelity(pool=pool)

        experiment_results = experiment_results.incumbent_trace(
            pool=pool,
            xaxis=xaxis,
            yaxis=yaxis,
        )

        # TODO: We should move to the new continuation fidelity metric.
        if xaxis == "end_time_since_global_start":
            experiment_results = experiment_results.rescale(
                xaxis=xaxis,
                by="max_fidelity",
                pool=pool,
            )

        if xrange is not None:
            experiment_results = experiment_results.in_range(
                bounds=xrange, xaxis=xaxis, pool=pool
            )

    print(f"[{now()}] Done! Duration {time.time() - starttime:.3f}...")

    for i, (benchmark, ax) in enumerate(zip(args.benchmarks, axs)):
        benchmark_results = experiment_results[benchmark]
        benchmark_config = experiment_results.benchmarks[benchmark]

        for algorithm in args.algorithms:
            algorithm_results = benchmark_results[algorithm]

            df = algorithm_results.df(index=xaxis, values=yaxis)

            # We fill in nans with any results seen before a given timestamp with
            # "ffill" and use "backfill" to fill in any NaNs that might occur at the
            # very start.
            df = df.fillna(method="ffill", axis=0).fillna(method="backfill", axis=0)

            is_last_row = lambda idx: idx >= (nrows - 1) * ncols
            is_first_column = lambda idx: idx % ncols == 0

            plot_incumbent(
                ax=ax,
                df=df,
                title=benchmark,
                xlabel=X_LABEL[xaxis] if is_last_row(i) else None,
                ylabel=Y_LABEL if is_first_column(i) else None,
                algorithm=algorithm,
                log_x=args.log_x,
                log_y=args.log_y,
                x_range=args.x_range,
                plot_default=benchmark_config.prior_error if args.plot_default else None,
                plot_optimum=benchmark_config.optimum if args.plot_optimum else None,
                plot_rs_10=benchmark_config.best_10_error if args.plot_rs_10 else None,
                plot_rs_25=benchmark_config.best_25_error if args.plot_rs_25 else None,
                plot_rs_100=benchmark_config.best_100_error if args.plot_rs_100 else None,
                force_prior_line="good" in benchmark_config.name,
            )

        if args.dynamic_y_lim:
            y_values = [getattr(result, yaxis) for result in benchmark_results.values()]
            y_min = min(y_values)
            y_max = max(y_values)

            plot_offset = 0.15
            dy = abs(y_max - y_min)
            ax.set_ylim(y_min - dy * plot_offset, y_max + dy * plot_offset)
        elif "jahs_colorectal_histology" in benchmark.name:
            ax.set_ylim(bottom=4.5, top=8)  # EDIT THIS IF JAHS COLORECTAL CHANGES
        else:
            ax.set_ylim(auto=True)

    sns.despine(fig)

    handles, labels = axs[0].get_legend_handles_labels()

    handles_to_plot, labels_to_plot = [], []
    handles_default, labels_default = [], []
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

    filename = args.filename
    if filename is None:
        filename = f"{args.experiment_group}_{args.plot_id}"

    if args.plot_max_fidelity_loss:
        plot_dir = plot_dir / "max_fidelity_loss"

    save_fig(
        fig,
        filename=filename,
        output_dir=plot_dir,
        extension=args.ext,
        dpi=args.dpi,
    )

    print(f"Plotting took {time.time() - starttime}")


if __name__ == "__main__":

    parser = get_parser()
    args = AttrDict(parser.parse_args().__dict__)

    if args.x_range is not None:
        assert len(args.x_range) == 2

    plot(args)
