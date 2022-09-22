import argparse
import errno
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from attrdict import AttrDict

from .configs.plotting.read_results import get_seed_info
from .configs.plotting.styles import X_LABEL, Y_LABEL
from .configs.plotting.utils import plot_incumbent, save_fig, set_general_plot_style

benchmark_configs_path = os.path.join(os.path.dirname(__file__), "configs/benchmark/")

map_axs = (
    lambda axs, idx, length: axs
    if length == 1
    else (axs[idx] if length == 2 else axs[idx // 2][idx % 2])
)


def plot(args):

    BASE_PATH = Path(".") if args.base_path is None else Path(args.base_path)

    set_general_plot_style()

    nrows = np.ceil(len(args.benchmarks) / 2).astype(int)
    ncols = 1 if len(args.benchmarks) == 1 else 2
    fig, axs = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(5.3, 2.2),
    )

    base_path = BASE_PATH / "results" / args.experiment_group
    output_dir = BASE_PATH / "plots" / args.experiment_group
    for benchmark_idx, benchmark in enumerate(args.benchmarks):
        _base_path = os.path.join(base_path, f"benchmark={benchmark}")
        if not os.path.isdir(_base_path):
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), _base_path)
        for algorithm in args.algorithms:
            _path = os.path.join(_base_path, f"algorithm={algorithm}")
            if not os.path.isdir(_path):
                raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), _path)

            incumbents = []
            costs = []

            for seed in sorted(os.listdir(_path)):
                # `algorithm` is passed to calculate continuation costs
                losses, infos, max_cost = get_seed_info(
                    _path, seed, algorithm=algorithm, cost_as_runtime=args.cost_as_runtime
                )
                incumbent = np.minimum.accumulate(losses)
                incumbents.append(incumbent)
                cost = [i["cost"] for i in infos]
                costs.append(cost)

            plot_incumbent(
                ax=map_axs(axs, benchmark_idx, len(args.benchmarks)),
                x=costs,
                y=incumbents,
                title=benchmark,
                xlabel=X_LABEL[args.cost_as_runtime],
                ylabel=Y_LABEL if benchmark_idx == 0 else None,
                algorithm=algorithm,
                log_x=args.log_x,
                log_y=args.log_y,
                # budget=args.budget,
                x_range=args.x_range,
                max_cost=None if args.cost_as_runtime else max_cost,
            )

    sns.despine(fig)

    handles, labels = map_axs(axs, 0, len(args.benchmarks)).get_legend_handles_labels()

    ncol_map = lambda n: 1 if n == 1 else (2 if n == 2 else int(np.ceil(n / 2)))
    ncol = ncol_map(len(args.algorithms))
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.25),
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
