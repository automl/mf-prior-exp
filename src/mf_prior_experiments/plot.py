import os
import errno
import argparse
import numpy as np
import seaborn as sns
from pathlib import Path
import matplotlib.pyplot as plt

from mf_prior_experiments.configs.plotting.read_results import (
    get_seed_info,
)
from mf_prior_experiments.configs.plotting.styles import (
    X_LABEL, Y_LABEL,
    X_MAP, Y_MAP
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
        for algorithm_idx, algorithm in enumerate(args.algorithms):
            _path = os.path.join(_base_path, f"algorithm={algorithm}")
            if not os.path.isdir(_path):
                raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), _path)

            incumbents = []
            costs = []

            for seed in sorted(os.listdir(_path)):
                losses, infos = get_seed_info(_path, seed, dataset=benchmark)
                losses = [-l for l in losses]  # TODO: confirm the return of benchmark
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
            )

            if len(args.benchmarks) > 1:
                axs[benchmark_idx].set_xticks(X_MAP)
                axs[benchmark_idx].set_xlim(min(X_MAP), max(X_MAP))
                axs[benchmark_idx].set_ylim(
                    min(Y_MAP[benchmark]),
                    max(Y_MAP[benchmark])
                )
            else:
                axs.set_xticks(X_MAP)
                axs.set_xlim(min(X_MAP), max(X_MAP))
                axs.set_ylim(
                    min(Y_MAP[benchmark]),
                    max(Y_MAP[benchmark])
                )

    sns.despine(fig)

    if len(args.benchmarks) > 1:
        handles, labels = axs[0].get_legend_handles_labels()
    else:
        handles, labels = axs.get_legend_handles_labels()

    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.15),
        ncol=len(args.algorithms) % 4,
        frameon=False
    )
    fig.tight_layout(pad=0, h_pad=.5)

    filename = args.filename
    if filename is None:
        filename = f"{args.experiment_group}_{args.plot_id}"
    save_fig(
        fig,
        filename=filename,
        output_dir=BASE_PATH / "plots"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="mf-prior-exp plotting",
    )
    parser.add_argument(
        "--base_path",
        type=str,
        default=None,
        help="path where `results/` exists"
    )
    parser.add_argument(
        "--experiment_group",
        type=str,
        default=""
    )
    parser.add_argument(
        "--benchmarks",
        nargs='+',
        default=["jahs_cifar10"]
    )
    parser.add_argument(
        "--algorithms",
        nargs='+',
        default=["random_search"]
    )
    parser.add_argument(
        "--plot_id",
        type=str,
        default="1"
    )
    parser.add_argument(
        "--filename",
        type=str,
        default=None,
        help="name out pdf file generated"
    )

    args = parser.parse_args()
    plot(args)  # pylint: disable=no-value-for-parameter
