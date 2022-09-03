import argparse
import errno
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from mf_prior_experiments.configs.plotting.read_results import (
    get_seed_info,
)
from mf_prior_experiments.configs.plotting.styles import (
    X_LABEL, Y_LABEL,
)
from mf_prior_experiments.configs.plotting.utils import (
    set_general_plot_style,
    plot_incumbent,
    save_fig
)


def plot(args):

    BASE_PATH = Path(".") if args.base_path is None else Path(args.base_path)

    set_general_plot_style()

    fig, axs = plt.subplots(
        nrows=1,
        ncols=len(args.benchmarks),
        figsize=(5.3, 2.2),
    )

    base_path = BASE_PATH / "results" / args.experiment_group
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
                log_y=args.log_y
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
    save_fig(fig, filename=filename, output_dir=BASE_PATH / "plots")


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
    parser.add_argument('--log_x', action='store_true')
    parser.add_argument('--log_y', action='store_true')
    parser.add_argument(
        "--filename", type=str, default=None, help="name out pdf file generated"
    )

    args = parser.parse_args()
    plot(args)  # pylint: disable=no-value-for-parameter
