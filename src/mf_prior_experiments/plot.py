from __future__ import annotations

import time
from itertools import chain, groupby, product, starmap
from multiprocessing import Pool
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from attrdict import AttrDict
from joblib import Parallel, delayed, parallel_backend
from more_itertools import all_equal, first

from .configs.plotting.read_results import (
    SINGLE_FIDELITY_ALGORITHMS,
    get_seed_info,
    load_yaml,
)
from .configs.plotting.styles import X_LABEL, Y_LABEL
from .configs.plotting.types import Algorithm, Benchmark, Trace
from .configs.plotting.utils import (
    get_max_fidelity,
    get_parser,
    interpolate_time,
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
    BASE_PATH: Path = DEFAULT_BASE_PATH if args.base_path is None else args.base_path
    BENCHMARK_CONFIG_DIR = BASE_PATH / "configs" / "benchmark"

    experiment_group: str = args.experiment_group
    plot_dir = BASE_PATH / "plots" / experiment_group

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

    algorithms = [Algorithm(a) for a in args.algorithms]
    benchmarks = [
        Benchmark.from_name(
            benchmark,
            config_dir=BENCHMARK_CONFIG_DIR,
        )
        for benchmark in args.benchmarks
    ]

    print(f"[{now()}] Processing ...")
    starttime = time.time()

    traces = Trace.load_all(
        base_path=BASE_PATH,
        experiment_group=experiment_group,
        benchmarks=benchmarks,
        algorithms=algorithms,
        parallel=args.parallel,
    )

    print(f"[{now()}] Done! Duration {time.time() - starttime:.3f}...")

    traces = [trace.with_continuations() for trace in traces]
    if args.n_workers <= 1:
        # fidelities: [1, 1, 3, 1, 9] -> [1, 2, 5, 6, 15]
        traces = [trace.with_cumulative_fidelity() for trace in traces]

    # TODO: We may want to change this with the new "continuation fidelity"
    # for parallel setup
    xaxis = "fidelity" if args.n_workers <= 1 else "end_time_since_global_start"
    yaxis = "max_fidelity_loss" if args.max_fidelity_loss else "loss"
    traces = [trace.incumbent_trace(xaxis=xaxis, yaxis=yaxis) for trace in traces]

    all_results: dict[tuple[Benchmark, Algorithm], list[Trace]] = {
        (bench, algo): sorted(trs, key=lambda r: r.seed)
        for (bench, algo), trs in groupby(
            traces, key=lambda t: (t.benchmark, t.algorithm)
        )
    }

    for i, (benchmark, ax) in enumerate(zip(benchmarks, axs)):

        for algorithm in algorithms:
            seed_traces: list[Trace] = all_results[(benchmark, algorithm)]

            df: pd.DataFrame = Trace.combine(seed_traces, xaxis=xaxis, yaxis=yaxis)

            # TODO: We should move to the new continuation fidelity metric.
            if xaxis == "end_time_since_global_start":
                df.index = df.index / benchmark.max_fidelity

            if xrange is not None:
                lower, upper = xrange
                df = df[(df.index >= lower) & (df.index <= upper)]  # type: ignore

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
                plot_default=benchmark.prior_error if args.plot_default else None,
                plot_optimum=benchmark.optimum if args.plot_optimum else None,
                plot_rs_10=benchmark.best_10_error if args.plot_rs_10 else None,
                plot_rs_25=benchmark.best_25_error if args.plot_rs_25 else None,
                plot_rs_100=benchmark.best_100_error if args.plot_rs_100 else None,
                force_prior_line="good" in benchmark.name,
            )

        if args.dynamic_y_lim:
            traces_ = chain.from_iterable(
                all_results[(benchmark, algo)] for algo in algorithms
            )
            results_ = chain.from_iterable(trace.results for trace in traces_)
            y_values = list(chain.from_iterable(getattr(r, yaxis) for r in results_))
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
        plot_dir = f"{plot_dir}/max_fidelity_loss/"

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
