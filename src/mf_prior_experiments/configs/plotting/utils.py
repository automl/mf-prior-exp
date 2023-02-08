from __future__ import annotations

from argparse import ArgumentParser, Namespace
from pathlib import Path

import matplotlib.pyplot as plt


def parse_args() -> Namespace:
    parser = ArgumentParser(description="mf-prior-exp plotting")

    parser.add_argument("--filename", type=str, default=None)
    parser.add_argument("--collect", action="store_true")
    parser.add_argument("--use-cache", action="store_true")

    parser.add_argument("--experiment_group", type=str, required=True)
    parser.add_argument("--algorithms", nargs="+", default=[])

    parser.add_argument("--benchmarks", nargs="+", default=[])
    parser.add_argument("--rr-good-corr-good-prior", nargs="+", default=[])
    parser.add_argument("--rr-good-corr-bad-prior", nargs="+", default=[])
    parser.add_argument("--rr-bad-corr-good-prior", nargs="+", default=[])
    parser.add_argument("--rr-bad-corr-bad-prior", nargs="+", default=[])

    parser.add_argument("--base_path", type=Path, default=None)
    parser.add_argument("--n_workers", type=int, default=1)
    parser.add_argument("--budget", nargs="+", type=float, default=None)
    parser.add_argument("--x_range", nargs=2, type=float, default=None)
    parser.add_argument("--x_together", type=float, default=None)

    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument("--ext", type=str, choices=["pdf", "png"], default="png")
    parser.add_argument("--plot_default", action="store_true")
    parser.add_argument("--plot_optimum", action="store_true")
    #parser.add_argument("--dynamic_y_lim", action="store_true")
    parser.add_argument("--parallel", action="store_true")

    args = parser.parse_args()

    if args.collect and args.use_cache:
        raise ValueError("Can't use --collect with --use-cache")

    if not args.collect and args.filename is None:
        raise ValueError("Must specify --filename unless using --collect")

    if args.x_together and args.x_range and args.x_together < args.x_range[0]:
        raise ValueError("--x_together must be larger than --x_range[0]")

    if not args.collect and len(args.algorithms) == 0:
        raise ValueError("Must specify --algorithms unless using --collect")

    if args.budget:
        raise ValueError("CD plots (which use --budget) not supported yet")

    benches = [
        args.rr_good_corr_good_prior,
        args.rr_good_corr_bad_prior,
        args.rr_bad_corr_good_prior,
        args.rr_bad_corr_bad_prior,
    ]
    if any(len(b) > 0 for b in benches) and not all(len(b) > 0 for b in benches):
        raise ValueError("Must specify all --rr args for relative rankings")

        #if not len(list(flatten(benches))) == len(set(flatten(benches))):
        #raise ValueError("Benchmarks in --rr must all be unique\n")

    return args

def set_general_plot_style():
    """
    sns.set_style("ticks")
    sns.set_context("paper")
    sns.set_palette("deep")
    """
    # plt.switch_backend("pgf")
    plt.rcParams.update(
        {
            "text.usetex": False,  # True,
            # "pgf.texsystem": "pdflatex",
            # "pgf.rcfonts": False,
            # "font.family": "serif",
            # "font.serif": [],
            # "font.sans-serif": [],
            # "font.monospace": [],
            "font.size": "10.90",
            "legend.fontsize": "9.90",
            "xtick.labelsize": "small",
            "ytick.labelsize": "small",
            "legend.title_fontsize": "small",
            # "bottomlabel.weight": "normal",
            # "toplabel.weight": "normal",
            # "leftlabel.weight": "normal",
            # "tick.labelweight": "normal",
            # "title.weight": "normal",
            # "pgf.preamble": r"""
            #    \usepackage[T1]{fontenc}
            #    \usepackage[utf8x]{inputenc}
            #    \usepackage{microtype}
            # """,
        }
    )
