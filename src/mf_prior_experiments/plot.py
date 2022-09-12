import argparse
import errno
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import yaml  # type: ignore
from attrdict import AttrDict

from .configs.plotting.read_results import get_seed_info
from .configs.plotting.styles import X_LABEL, Y_LABEL
from .configs.plotting.utils import plot_incumbent, save_fig, set_general_plot_style

benchmark_configs_path = os.path.join(os.path.dirname(__file__), "configs/benchmark/")


def plot(args):

    BASE_PATH = Path(".") if args.base_path is None else Path(args.base_path)

    set_general_plot_style()

    fig, axs = plt.subplots(
        nrows=1,
        ncols=len(args.benchmarks),
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
                losses, infos = get_seed_info(_path, seed, algorithm=algorithm)
                incumbent = np.minimum.accumulate(losses)
                incumbents.append(incumbent)
                cost = [i["cost"] for i in infos]
                costs.append(cost)

            plot_incumbent(
                ax=axs[benchmark_idx] if len(args.benchmarks) > 1 else axs,
                x=costs,
                y=incumbents,
                title=benchmark,
                xlabel=X_LABEL,
                ylabel=Y_LABEL if benchmark_idx == 0 else None,
                algorithm=algorithm,
                log_x=args.log_x,
                log_y=args.log_y,
                budget=args.budget,
            )

    sns.despine(fig)

    if len(args.benchmarks) > 1:
        handles, labels = axs[0].get_legend_handles_labels()
    else:
        handles, labels = axs.get_legend_handles_labels()

    ncol = int(np.ceil(len(args.algorithms) / 2))
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.15),
        ncol=len(args.algorithms) if not ncol else ncol,
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
    parser.add_argument("--benchmarks", nargs="+", default=["jahs_cifar10"])
    parser.add_argument("--algorithms", nargs="+", default=["random_search"])
    parser.add_argument("--plot_id", type=str, default="1")
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

    args = AttrDict(parser.parse_args().__dict__)
    budget = None
    # reading benchmark budget if only one benchmark is being plotted
    if len(args.benchmarks) == 1:
        with open(
            os.path.join(benchmark_configs_path, f"{args.benchmarks[0]}.yaml"),
            encoding="utf-8",
        ) as f:
            _args = AttrDict(yaml.load(f, yaml.Loader))
            if "budget" in _args:
                budget = _args.budget
    # TODO: make log scaling of plots also a feature of the benchmark
    args.update({"budget": budget})
    plot(args)  # pylint: disable=no-value-for-parameter
