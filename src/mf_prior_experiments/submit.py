""" Submits to slurm using a hydra API.

"""
import argparse
import itertools
import os
from pathlib import Path


def parse_argument_string(args):
    def get_argument_settings(argument):
        """Transforms 'a=1,2,3' to ('a=1', 'a=2', 'a=3')."""
        name, values = argument.split("=")
        if values.startswith("range("):
            range_params = values[len("range(") : -1]
            range_params = (int(param) for param in range_params.split(", "))
            return [f"{name}={value}" for value in range(*range_params)]
        else:
            return [f"{name}={value}" for value in values.split(",")]

    def get_all_argument_settings(arguments):
        """
        Transforms ['a=1,2,3', 'b=1'] to [('a=1', 'b=1'), ('a=2', 'b=1'), ('a=3', 'b=1')]
        """
        return itertools.product(
            *(get_argument_settings(argument) for argument in arguments)
        )

    def get_all_argument_strings(argument_settings):
        """Transforms [('a=1', 'b=1')] to ('a=1 b=1',)"""
        return (" ".join(argument_setting) for argument_setting in argument_settings)

    argument_settings = get_all_argument_settings(args.arguments)
    argument_strings = list(get_all_argument_strings(argument_settings))
    argument_string = "\n".join(argument_strings)
    argument_final_string = f"ARGS=(\n{argument_string}\n)"

    return argument_final_string, len(argument_strings)


def construct_script(args, cluster_oe_dir):
    argument_string, num_tasks = parse_argument_string(args)

    script = list()
    script.append("#!/bin/bash")
    script.append(f"#SBATCH --time {args.time}")
    script.append(f"#SBATCH --job-name {args.job_name}")
    script.append(f"#SBATCH --partition {args.partition}")
    script.append(f"#SBATCH --array 0-{num_tasks - 1}%{args.max_tasks}")
    script.append(f"#SBATCH --error {cluster_oe_dir}/%N_%A_%x_%a.oe")
    script.append(f"#SBATCH --output {cluster_oe_dir}/%N_%A_%x_%a.oe")
    script.append(f"#SBATCH --mem-per-cpu {args.memory}")
    if args.n_worker > 1:
        script.append(f"#SBATCH -c {args.n_worker}")
    if args.exclude:
        script.append(f"#SBATCH --exclude {args.exclude}")
    script.append("")
    script.append(argument_string)
    script.append("")
    if args.n_worker > 1:
        script.append(
            f"srun --ntasks {args.n_worker} --cpus-per-task 1 "
            f"python -m mf_prior_experiments.run experiment_group={args.experiment_group} "
            f"${{ARGS[@]:{len(args.arguments)}*$SLURM_ARRAY_TASK_ID:{len(args.arguments)}}}"
        )
    else:
        script.append(
            f"python -m mf_prior_experiments.run experiment_group={args.experiment_group} "
            f"${{ARGS[@]:{len(args.arguments)}*$SLURM_ARRAY_TASK_ID:{len(args.arguments)}}}"
        )
    return "\n".join(script) + "\n"  # type: ignore[assignment]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_group", default="test")
    parser.add_argument("--max_tasks", default=100, type=int)
    parser.add_argument("--time", default="0-23:59")
    parser.add_argument("--job_name", default="test")
    parser.add_argument("--memory", default=0, type=int)
    parser.add_argument("--n_worker", default=1, type=int)
    parser.add_argument("--partition", required=True)
    parser.add_argument("--arguments", nargs="+")
    parser.add_argument(
        "--exclude", default=None, type=str, help="example: kisexe24,kisexe34"
    )
    args = parser.parse_args()

    experiment_group_dir = Path("results", args.experiment_group)
    cluster_oe_dir = Path(experiment_group_dir, ".cluster_oe")
    scripts_dir = Path(experiment_group_dir, ".submit")

    experiment_group_dir.mkdir(parents=True, exist_ok=True)
    cluster_oe_dir.mkdir(parents=True, exist_ok=True)
    scripts_dir.mkdir(parents=True, exist_ok=True)

    script = construct_script(args, cluster_oe_dir)

    num_scripts = len(list(scripts_dir.glob("*.sh")))
    script_path = Path(scripts_dir, f"{num_scripts}.sh")
    submission_commmand = f"sbatch {script_path}"
    print(f"Running {submission_commmand} with script:\n\n{script}")
    if input("Ok? [Y|n] -- ").lower() in {"y", ""}:
        script_path.write_text(script, encoding="utf-8")  # type: ignore[arg-type]
        os.system(submission_commmand)
    else:
        print("Not submitting.")
