from __future__ import annotations

from pathlib import Path
from typing import Any, Iterator

import mfpbench
from mfpbench import JAHSBenchmark, PD1Benchmark, YAHPOBenchmark

SEED = 1
N_SAMPLES = 25
EPSILON = 1e-3
ITERATIONS_MAX = 1_000


JOBFILE = Path("submit_calculate.sh")
PARTITION = "bosch_cpu-cascadelake"
CPUS = 1
JOB_TIME = ITERATIONS_MAX + (5 * 60)  # Max iteration count with 5 min extra
MEMORY = 25_000

HERE = Path(__file__).absolute().resolve().parent
SCRIPT_PATH = HERE.parent / "src" / "mf-prior-bench" / "correlations.py"
DATADIR = HERE.parent / "data"
LOGDIR = HERE / "logs"
RESULTSDIR = HERE / "results"

JAHSDATA_DIR = DATADIR / "jahs-bench-data"
PD1DATA_DIR = DATADIR / "pd1-data"
YAHPODATA_DIR = DATADIR / "yahpo-gym-data"


def benchmarks(
    seed: int = SEED,
    only: list[str] | None = None,
    exclude: list[str] | None = None,
) -> Iterator[tuple[str, dict[str, Any]]]:
    for name, cls in mfpbench._mapping.items():
        if only and not any(o in name for o in only):
            continue

        if exclude and any(e in name for e in exclude):
            continue

        if cls.has_conditionals:
            continue

        if issubclass(cls, YAHPOBenchmark) and cls.instances is not None:
            for task_id in cls.instances:
                yield f"{name}-{task_id}", dict(
                    benchmark=name,
                    task_id=task_id,
                    seed=seed,
                    datadir=YAHPODATA_DIR,
                )
        elif issubclass(cls, PD1Benchmark):
            yield name, dict(benchmark=name, seed=seed, datadir=PD1DATA_DIR)
        elif issubclass(cls, JAHSBenchmark):
            yield name, dict(benchmark=name, seed=seed, datadir=JAHSDATA_DIR)
        else:
            yield name, dict(benchmark=name, seed=seed)


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

    if not RESULTSDIR.exists():
        RESULTSDIR.mkdir(exist_ok=True)

    if not LOGDIR.exists():
        LOGDIR.mkdir(exist_ok=True)

    blocks: list[str] = []

    for i, (name, kwargs) in enumerate(benchmarks(exclude=["iaml", "rbv2"])):
        cmd = " ".join(
            [
                f"python {SCRIPT_PATH}",
                f'--name="{name}"',
                f"--n_samples={N_SAMPLES}",
                f"--epsilon={EPSILON}",
                f"--iterations_max={ITERATIONS_MAX}",
                f"--results_dir={RESULTSDIR}",
            ]
            + [f'--{k}="{v}"' for k, v in kwargs.items()]
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
        "cpus-per-task": CPUS,
        "time": timestr(JOB_TIME),
        "mem": str(MEMORY),
        "partition": PARTITION,
        "array": ",".join([str(i) for i in range(len(blocks))]),
        "error": LOGDIR / "error_%a.txt",
        "out": LOGDIR / "out_%a.txt",
        "job-name": "correlations-mfpbench",
    }
    sbatch_args = "\n".join([f"#SBATCH --{k}={v}" for k, v in kw.items()])

    with JOBFILE.open("w") as f:
        f.writelines(["#!/bin/sh", "\n", *sbatch_args, "\n", *blocks])

    print("Done")


if __name__ == "__main__":
    create_script()
