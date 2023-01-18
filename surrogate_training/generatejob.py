from __future__ import annotations

from itertools import product
from pathlib import Path

ALLOWED_DATASETS = [
    "imagenet-resnet-512",
    "cifar100-wide_resnet-2048",
    "lm1b-transformer-2048",
    "uniref50-transformer-128",
    "translatewmt-xformer-64",
]

SEED = 1
OPT_TIME_SECONDS = int(4 * 60 * 60)
CV_FOLDS = 5
DEHB_WORKERS = 8

HERE = Path(__file__).absolute().resolve().parent
DATADIR = HERE.parent / "data"
LOGDIR = HERE / "logs"
PD1DATA_DIR = DATADIR / "pd1-data"
SURROGATE_DIR = PD1DATA_DIR / "surrogates"

JOBFILE = Path("submit.sh")
PARTITION = "bosch_cpu-cascadelake"
JOB_TIME = (OPT_TIME_SECONDS * 2) + (5 * 60)  # Double time with 5 min extra
MEMORY = 32_000

datasets = [
    f.name.replace("_surrogate.csv", "")
    for f in PD1DATA_DIR.iterdir()
    if not f.is_dir() and f.suffix == ".csv" and "tabular" not in f.name
]
datasets = [d for d in datasets if d in ALLOWED_DATASETS]

# Some also have a test_error_rate but we leave it out for now
# as we only select the ones that are known
dataset_metrics = list(product(datasets, ["valid_error_rate", "train_cost"]))


def surrogate_path(dataset: str, metric: str) -> Path:
    return SURROGATE_DIR / f"{dataset}-{metric}.json"


def timestr(seconds: int) -> str:
    days = int(seconds / (24 * 60 * 60))
    seconds -= days * (24 * 60 * 60)

    hours = int(seconds / (60 * 60))
    seconds -= hours * (60 * 60)

    mins = int(seconds / (60))
    seconds -= mins * (60)

    seconds = int(seconds)

    return f"{days}-{hours:02}:{mins:02}:{seconds:02}"


def create_script() -> None:
    blocks: list[str] = []
    # needed = [(d, m) for d, m in dataset_metrics if not surrogate_path(d, m).exists()]
    needed = dataset_metrics
    for i, (d, m) in enumerate(needed):
        cmd = " ".join(
            [
                "python -m mfpbench.pd1.surrogate.training",
                f"--workers={DEHB_WORKERS}",
                f"--time={OPT_TIME_SECONDS}",
                f"--seed={SEED}",
                f"--cv={CV_FOLDS}",
                f"--datadir={DATADIR}",
                f'--dataset="{d}"',
                f'--y="{m}"',
                f'--dehb-output="dehb_output_{d}-{m}"',
            ]
        )
        block = "\n".join(
            [
                "\n",
                f"if [ $SLURM_ARRAY_TASK_ID -eq {i} ]; then",
                f"    {cmd}",
                "fi",
                "\n",
            ]
        )
        blocks.append(block)

    kw = {
        "export": "ALL",
        "cpus-per-task": DEHB_WORKERS,
        "time": timestr(JOB_TIME),
        "mem": str(MEMORY),
        "partition": PARTITION,
        "array": ",".join([str(i) for i in range(len(blocks))]),
        "error": LOGDIR / "error_%a.txt",
        "out": LOGDIR / "out_%a.txt",
        "job-name": "surrogates-mfpbench",
    }
    sbatch_args = "\n".join([f"#SBATCH --{k}={v}" for k, v in kw.items()])

    LOGDIR.mkdir(exist_ok=True)
    SURROGATE_DIR.mkdir(exist_ok=True)

    with JOBFILE.open("w") as f:
        f.writelines(["#!/bin/sh", "\n", *sbatch_args, "\n", *blocks])

    print("Done")


if __name__ == "__main__":
    create_script()
