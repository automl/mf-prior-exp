import errno
import os
import time
from multiprocessing import Manager
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from attrdict import AttrDict
from joblib import Parallel, delayed, parallel_backend

from .configs.plotting.read_results import get_seed_info, load_yaml
from .configs.plotting.styles import X_LABEL, Y_LABEL
from .configs.plotting.utils import (
    get_parser,
    interpolate_time,
    plot_incumbent,
    save_fig,
    set_general_plot_style,
)

benchmark_configs_path = os.path.join(os.path.dirname(__file__), "configs/benchmark/")

map_axs = (
    lambda axs, idx, length, ncols: axs
    if length == 1
    else (axs[idx] if length == ncols else axs[idx // ncols][idx % ncols])
)


def _process_seed(
    _path, seed, algorithm, key_to_extract, cost_as_runtime, results, n_workers
):
    print(
        f"[{time.strftime('%H:%M:%S', time.localtime())}] "
        f"[-] [{algorithm}] Processing seed {seed}..."
    )
    try:
        # `algorithm` is passed to calculate continuation costs
        losses, infos, max_cost = get_seed_info(
            _path,
            seed,
            algorithm=algorithm,
            cost_as_runtime=cost_as_runtime,
            n_workers=n_workers,
        )
        incumbent = np.minimum.accumulate(losses)
        cost = [i[key_to_extract] for i in infos]
        results["incumbents"].append(incumbent)
        results["costs"].append(cost)
        results["max_costs"].append(max_cost)
    except Exception as e:
        print(repr(e))
        print(f"Seed {seed} did not work from {_path}/{algorithm}")


def plot(args):

    starttime = time.time()

    BASE_PATH = (
        Path(__file__).parent / "../.."
        if args.base_path is None
        else Path(args.base_path)
    )

    KEY_TO_EXTRACT = "cost" if args.cost_as_runtime else "fidelity"

    set_general_plot_style()

    if args.research_question == 1:
        ncols = 1 if len(args.benchmarks) == 1 else 2
        # ncol_map = lambda n: 1 if n == 1 else (2 if n == 2 else int(np.ceil(n / 2)))

        legend_ncol = len(args.algorithms)
        legend_ncol += 1 if args.plot_default is not None else 0
        legend_ncol += 1 if args.plot_optimum is not None else 0
    elif args.research_question == 2:
        if args.benchmarks is None:
            args.benchmarks = [
                f"jahs_cifar10_prior-{args.which_prior}",
                f"jahs_fashion_mnist_prior-{args.which_prior}",
                #                f"jahs_colorectal_histology_prior-{args.which_prior}",
                f"lcbench-189862_prior-{args.which_prior}",
                f"lcbench-189866_prior-{args.which_prior}",
                f"translatewmt_xformer_64_prior-{args.which_prior}",
                f"lm1b_transformer_2048_prior-{args.which_prior}",
                #               f"uniref50_transformer_prior-{args.which_prior}",
            ]
        ncols = 4
        legend_ncol = len(args.algorithms)
        legend_ncol += 1 if args.plot_default is not None else 0
        legend_ncol += 1 if args.plot_optimum is not None else 0
    else:
        raise ValueError("Plotting works only for RQ1 and RQ2.")
    nrows = np.ceil(len(args.benchmarks) / ncols).astype(int)
    print("===============")
    print(nrows)
    print("===============")
    bbox_y_mapping = {
        1: -0.20,
        2: -0.11,
        3: -0.07,
        4: -0.05,
        5: -0.04,
    }
    bbox_to_anchor = (0.5, bbox_y_mapping[nrows])
    figsize = (4 * ncols, 3 * nrows)

    fig, axs = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=figsize,
    )

    base_path = BASE_PATH / "results" / args.experiment_group
    output_dir = BASE_PATH / "plots" / args.experiment_group
    print(
        f"[{time.strftime('%H:%M:%S', time.localtime())}]"
        f" Processing {len(args.benchmarks)} benchmarks "
        f"and {len(args.algorithms)} algorithms..."
    )

    for benchmark_idx, benchmark in enumerate(args.benchmarks):
        print(
            f"[{time.strftime('%H:%M:%S', time.localtime())}] "
            f"[{benchmark_idx}] Processing {benchmark} benchmark..."
        )
        benchmark_starttime = time.time()
        # loading the benchmark yaml
        _bench_spec_path = (
            BASE_PATH
            / "src"
            / "mf_prior_experiments"
            / "configs"
            / "benchmark"
            / f"{benchmark}.yaml"
        )
        plot_default = None
        if args.plot_default and os.path.isfile(_bench_spec_path):
            try:
                plot_default = load_yaml(_bench_spec_path).prior_highest_fidelity_error
            except Exception as e:
                print(repr(e))

                print(f"Could not load error for benchmark yaml {_bench_spec_path}")

        plot_optimum = None
        if args.plot_optimum and os.path.isfile(_bench_spec_path):
            try:
                plot_optimum = load_yaml(_bench_spec_path).optimum
            except Exception as e:
                print(repr(e))
                print(f"Could not load optimum for benchmark yaml {_bench_spec_path}")

        plot_rs_10 = None
        if args.plot_rs_10 and os.path.isfile(_bench_spec_path):
            try:
                plot_rs_10 = load_yaml(_bench_spec_path).best_10_error
            except Exception as e:
                print(repr(e))
                print(f"Could not load optimum for benchmark yaml {_bench_spec_path}")

        plot_rs_25 = None
        if args.plot_rs_25 and os.path.isfile(_bench_spec_path):
            try:
                plot_rs_25 = load_yaml(_bench_spec_path).best_25_error
            except Exception as e:
                print(repr(e))
                print(f"Could not load optimum for benchmark yaml {_bench_spec_path}")

        plot_rs_100 = None
        if args.plot_rs_100 and os.path.isfile(_bench_spec_path):
            try:
                plot_rs_100 = load_yaml(_bench_spec_path).best_100_error
            except Exception as e:
                print(repr(e))
                print(f"Could not load optimum for benchmark yaml {_bench_spec_path}")

        _base_path = os.path.join(base_path, f"benchmark={benchmark}")
        if not os.path.isdir(_base_path):
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), _base_path)

        y_max = []
        y_min = None
        if args.dynamic_y_lim:
            for algorithm in ["random_search", "random_search_prior-default-first"]:
                _path = os.path.join(_base_path, f"algorithm={algorithm}")
                if not os.path.isdir(_path):
                    raise FileNotFoundError(
                        errno.ENOENT, os.strerror(errno.ENOENT), _path
                    )
                seeds = sorted(os.listdir(_path))

                if args.parallel:
                    manager = Manager()
                    results = manager.dict(
                        incumbents=manager.list(),
                        costs=manager.list(),
                        max_costs=manager.list(),
                    )
                    with parallel_backend(args.parallel_backend, n_jobs=-1):
                        Parallel()(
                            delayed(_process_seed)(
                                _path,
                                seed,
                                algorithm,
                                KEY_TO_EXTRACT,
                                args.cost_as_runtime,
                                results,
                                args.n_workers,
                            )
                            for seed in seeds
                        )

                else:
                    results = dict(incumbents=[], costs=[], max_costs=[])
                    # pylint: disable=expression-not-assigned
                    [
                        _process_seed(
                            _path,
                            seed,
                            algorithm,
                            KEY_TO_EXTRACT,
                            args.cost_as_runtime,
                            results,
                            args.n_workers,
                        )
                        for seed in seeds
                    ]

                y_max.extend([min(r[:2]) for r in results["incumbents"][:]])
            y_max = np.mean(y_max)

        for algorithm in args.algorithms:
            _path = os.path.join(_base_path, f"algorithm={algorithm}")
            if not os.path.isdir(_path):
                raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), _path)

            algorithm_starttime = time.time()
            seeds = sorted(os.listdir(_path))

            if args.parallel:
                manager = Manager()
                results = manager.dict(
                    incumbents=manager.list(),
                    costs=manager.list(),
                    max_costs=manager.list(),
                )
                with parallel_backend(args.parallel_backend, n_jobs=-1):
                    Parallel()(
                        delayed(_process_seed)(
                            _path,
                            seed,
                            algorithm,
                            KEY_TO_EXTRACT,
                            args.cost_as_runtime,
                            results,
                            args.n_workers,
                        )
                        for seed in seeds
                    )

            else:
                results = dict(incumbents=[], costs=[], max_costs=[])
                # pylint: disable=expression-not-assigned
                [
                    _process_seed(
                        _path,
                        seed,
                        algorithm,
                        KEY_TO_EXTRACT,
                        args.cost_as_runtime,
                        results,
                        args.n_workers,
                    )
                    for seed in seeds
                ]

            print(f"Time to process algorithm data: {time.time() - algorithm_starttime}")

            ax = map_axs(axs, benchmark_idx, len(args.benchmarks), ncols)
            x = results["costs"][:]
            y = results["incumbents"][:]
            max_cost = None if args.cost_as_runtime else max(results["max_costs"][:])

            if isinstance(x, list):
                x = np.array(x)
            if isinstance(y, list):
                y = np.array(y)

            df = interpolate_time(
                incumbents=y, costs=x, x_range=args.x_range, scale_x=max_cost
            )

            import pandas as pd

            x_max = np.inf if args.x_range is None else int(args.x_range[-1])
            new_entry = {c: np.nan for c in df.columns}
            _df = pd.DataFrame.from_dict(new_entry, orient="index").T
            _df.index = [x_max]
            df = pd.concat((df, _df)).sort_index()
            df = df.fillna(method="backfill", axis=0).fillna(method="ffill", axis=0)

            y_min = min(
                list(
                    filter(
                        None,
                        [
                            np.mean(df.query(f"index <= {x_max}").values[-1]),
                            y_min,
                            plot_default,
                            plot_optimum,
                        ],
                    )
                )
            )
            is_last_row = lambda idx: idx >= (nrows - 1) * ncols
            # pylint: disable=cell-var-from-loop
            is_first_column = lambda idx: benchmark_idx % ncols == 0
            plot_incumbent(
                ax=ax,
                # x=x,
                # y=y,
                df=df,
                title=benchmark,
                xlabel=X_LABEL[args.cost_as_runtime]
                if is_last_row(benchmark_idx)
                else None,
                ylabel=Y_LABEL if is_first_column(benchmark_idx) else None,
                algorithm=algorithm,
                log_x=args.log_x,
                log_y=args.log_y,
                x_range=args.x_range,
                # max_cost=max_cost,
                plot_default=plot_default,
                plot_optimum=plot_optimum,
                plot_rs_10=plot_rs_10,
                plot_rs_25=plot_rs_25,
                plot_rs_100=plot_rs_100,
                force_prior_line="good" in benchmark,
            )
            if args.dynamic_y_lim:
                plot_offset = 0.15
                dy = abs(y_max - y_min)
                ax.set_ylim(y_min - dy * plot_offset, y_max + dy * plot_offset)
            elif "jahs_colorectal_histology" in benchmark:
                # EDIT THIS IF JAHS COLORECTAL CHANGES
                ax.set_ylim(bottom=4.5, top=8)
            else:
                ax.set_ylim(auto=True)

            print(f"Time to plot algorithm data: {time.time() - algorithm_starttime}")
        print(f"Time to process benchmark data: {time.time() - benchmark_starttime}")

    sns.despine(fig)

    handles, labels = map_axs(
        axs, 0, len(args.benchmarks), ncols
    ).get_legend_handles_labels()

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
    save_fig(
        fig,
        filename=filename,
        output_dir=output_dir,
        extension=args.ext,
        dpi=args.dpi,
    )

    print(f"Plotting took {time.time() - starttime}")


if __name__ == "__main__":

    parser = get_parser()
    args = AttrDict(parser.parse_args().__dict__)

    if args.x_range is not None:
        assert len(args.x_range) == 2

    # budget = None
    # # reading benchmark budget if only one benchmark is being plotted
    # if len(args.benchmarks) == 1:
    #     with open(
    #         os.path.join(benchmark_configs_path, f"{args.benchmarks[0]}.yaml"),
    #         encoding="utf-8",
    #     ) as f:
    #         _args = AttrDict(yaml.load(f, yaml.Loader))
    #         if "budget" in _args:
    #             budget = _args.budget
    # # TODO: make log scaling of plots also a feature of the benchmark
    # args.update({"budget": budget})
    plot(args)  # pylint: disable=no-value-for-parameter
