import argparse
import errno
import os
from multiprocessing import Manager
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from attrdict import AttrDict
from joblib import Parallel, delayed, parallel_backend

from .configs.plotting.read_results import get_seed_info, load_yaml
from .configs.plotting.styles import X_LABEL, Y_LABEL
from .configs.plotting.utils import plot_incumbent, save_fig, set_general_plot_style

benchmark_configs_path = os.path.join(os.path.dirname(__file__), "configs/benchmark/")

map_axs = (
    lambda axs, idx, length: axs
    if length == 1
    else (axs[idx] if length == 2 else axs[idx // 2][idx % 2])
)


def plot(args):

    import time

    start_time = time.time()

    BASE_PATH = (
        Path(__file__).parent / "../.."
        if args.base_path is None
        else Path(args.base_path)
    )

    KEY_TO_EXTRACT = "cost" if args.cost_as_runtime else "fidelity"

    set_general_plot_style()

    nrows = np.ceil(len(args.benchmarks) / 2).astype(int)
    ncols = 1 if len(args.benchmarks) == 1 else 2
    fig, axs = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(10.3, 6.2),
    )

    base_path = BASE_PATH / "results" / args.experiment_group
    output_dir = BASE_PATH / "plots" / args.experiment_group
    for benchmark_idx, benchmark in enumerate(args.benchmarks):
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

        if args.plot_optimum and os.path.isfile(_bench_spec_path):
            try:
                plot_default = load_yaml(_bench_spec_path).optimum
            except Exception as e:
                print(repr(e))
                print(f"Could not load optimum for benchmark yaml {_bench_spec_path}")

        _base_path = os.path.join(base_path, f"benchmark={benchmark}")
        if not os.path.isdir(_base_path):
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), _base_path)
        for algorithm in args.algorithms:
            _path = os.path.join(_base_path, f"algorithm={algorithm}")
            if not os.path.isdir(_path):
                raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), _path)

            manager = Manager()
            results = manager.dict(
                incumbents=manager.list(), costs=manager.list(), max_costs=manager.list()
            )

            def _process_seed(_path, seed, algorithm, cost_as_runtime, results):
                # `algorithm` is passed to calculate continuation costs
                losses, infos, max_cost = get_seed_info(
                    _path, seed, algorithm=algorithm, cost_as_runtime=cost_as_runtime
                )
                incumbent = np.minimum.accumulate(losses)
                cost = [i[KEY_TO_EXTRACT] for i in infos]
                results["incumbents"].append(incumbent)
                results["costs"].append(cost)
                results["max_costs"].append(max_cost)

            seeds = sorted(os.listdir(_path))
            with parallel_backend("threading", n_jobs=-1):
                Parallel()(
                    delayed(_process_seed)(
                        _path, seed, algorithm, args.cost_as_runtime, results
                    )
                    for seed in seeds
                )

            plot_incumbent(
                ax=map_axs(axs, benchmark_idx, len(args.benchmarks)),
                x=results["costs"][:],
                y=results["incumbents"][:],
                title=benchmark,
                xlabel=X_LABEL[args.cost_as_runtime],
                ylabel=Y_LABEL if benchmark_idx == 0 else None,
                algorithm=algorithm,
                log_x=args.log_x,
                log_y=args.log_y,
                x_range=args.x_range,
                max_cost=None if args.cost_as_runtime else max(results["max_costs"][:]),
                plot_default=plot_default,
            )

    sns.despine(fig)

    handles, labels = map_axs(axs, 0, len(args.benchmarks)).get_legend_handles_labels()

    ncol_map = lambda n: 1 if n == 1 else (2 if n == 2 else int(np.ceil(n / 2)))
    ncol = ncol_map(len(args.algorithms))
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.1),
        ncol=ncol,
        frameon=True,
    )
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

    print(time.time() - start_time)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="mf-prior-exp plotting",
    )
    parser.add_argument(
        "--base_path", type=str, default=None, help="path where `results/` exists"
    )
    parser.add_argument("--experiment_group", type=str, default="")
    parser.add_argument("--benchmarks", nargs="+", default=None)
    parser.add_argument("--algorithms", nargs="+", default=None)
    parser.add_argument("--plot_id", type=str, default="1")
    parser.add_argument("--x_range", nargs="+", default=None, type=float)
    parser.add_argument("--log_x", action="store_true")
    parser.add_argument("--log_y", action="store_true")
    parser.add_argument(
        "--filename", type=str, default=None, help="name out pdf file generated"
    )
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument(
        "--ext",
        type=str,
        choices=["pdf", "png"],
        default="pdf",
        help="the file extension or the plot file type",
    )
    parser.add_argument(
        "--cost_as_runtime",
        default=False,
        action="store_true",
        help="Default behaviour to use fidelities on the x-axis. "
        "This parameter uses the training cost/runtime on the x-axis",
    )
    parser.add_argument(
        "--plot_default",
        default=False,
        action="store_true",
        help="plots a horizontal line for the prior score if available",
    )
    parser.add_argument(
        "--plot_optimum",
        default=False,
        action="store_true",
        help="plots a horizontal line for the optimum score if available",
    )

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
