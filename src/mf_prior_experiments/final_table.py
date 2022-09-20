import argparse
import errno
import os

import numpy as np
import pandas as pd
from attrdict import AttrDict
from path import Path
from scipy import stats

from .configs.plotting.read_results import get_seed_info

benchmark_configs_path = os.path.join(os.path.dirname(__file__), "configs/benchmark/")


def plot(args):
    BASE_PATH = Path(".") if args.base_path is None else Path(args.base_path)

    base_path = BASE_PATH / "results" / args.experiment_group
    output_dir = BASE_PATH / "tables" / args.experiment_group
    final_table = dict()
    for _, benchmark in enumerate(args.benchmarks):
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

            from .configs.plotting.utils import interpolate_time

            incumbents = np.array(incumbents)
            costs = np.array(costs)

            df = interpolate_time(incumbents, costs)

            if args.budget is not None:
                df = df.query(f"index <= {args.budget}")
            final_mean = df.mean(axis=1).values[-1]
            final_std_error = stats.sem(df.values, axis=1)[-1]

            if benchmark not in final_table:
                final_table[benchmark] = dict()
            final_table[benchmark][
                algorithm
            ] = fr"${np.round(final_mean, 2)} \pm {np.round(final_std_error, 2)}$"

    final_table = pd.DataFrame.from_dict(final_table, orient="index")

    filename = args.filename
    if filename is None:
        filename = f"{args.experiment_group}_{args.plot_id}"

    output_dir = Path(output_dir)
    output_dir.makedirs_p()

    with open(
        os.path.join(output_dir, f"{filename}.tex"),
        "w",
        encoding="utf-8",
    ) as f:
        f.write("\\begin{table}[htbp]" + " \n")
        f.write("\\centering" + " \n")

        f.write(
            "\\begin{tabular}{"
            + " | ".join(["c"] * (len(final_table.columns) + 1))
            + "}\n"
        )
        f.write("\\toprule" + " \n")
        f.write("{} ")

        import re

        for c in final_table.columns:
            f.write("& ")
            f.write(re.sub("_", r"\\_", c) + " ")
        f.write("\\\\\n")
        f.write("\\midrule" + " \n")

        for i, row in final_table.iterrows():
            f.write(re.sub("_", r"\\_", str(row.name)) + " ")
            f.write(" & " + " & ".join([str(x) for x in row.values]))
            f.write(" \\\\\n")
        f.write("\\bottomrule" + " \n")
        f.write("\\end{tabular}" + " \n")
        f.write("\\caption{" f"{args.caption}" + "}" + " \n")
        f.write("\\label{" f"{args.label}" + "}" + " \n")
        f.write("\\end{table}")
        # f.write(final_table.to_latex())
    print(f"{final_table}")
    print(f'Saved to "{output_dir}/{filename}.tex"')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="mf-prior-exp plotting",
    )
    parser.add_argument(
        "--base_path", type=str, default=None, help="path where `results/` exists"
    )
    parser.add_argument("--experiment_group", type=str, default="")
    parser.add_argument("--caption", type=str, default="TODO")
    parser.add_argument("--label", type=str, default="TODO")
    parser.add_argument("--budget", type=float, default=None)
    parser.add_argument("--benchmarks", nargs="+", default=["jahs_cifar10"])
    parser.add_argument("--algorithms", nargs="+", default=["random_search"])
    parser.add_argument("--plot_id", type=str, default="1")
    parser.add_argument(
        "--filename", type=str, default=None, help="name out pdf file generated"
    )
    args = AttrDict(parser.parse_args().__dict__)
    plot(args)  # pylint: disable=no-value-for-parameter
