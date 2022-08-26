"""Generates all possible benchmark configs from mfpbench."""
from __future__ import annotations

from pathlib import Path

import mfpbench
import yaml
from mfpbench import (
    Benchmark,
    JAHSBenchmark,
    LCBenchBenchmark,
    MFHartmann3Benchmark,
    MFHartmann6Benchmark,
    YAHPOBenchmark,
)

HERE = Path(__file__).parent.resolve()

dir_lookup = {JAHSBenchmark: "jahs-bench-data", YAHPOBenchmark: "yahpo-gym-data"}


def datadir(cls: type[Benchmark]) -> str | None:
    """Get the datadir arg for a benchmark.

    Parameters
    ----------
    cls : type[Benchmark]
        The type of the benchhmark

    Returns
    -------
    str | None
        Returns the ``datadir`` arg if applicable, else None
    """
    folder = next((v for k, v in dir_lookup.items() if issubclass(cls, k)), None)
    if folder is None:
        return None
    else:
        return "${hydra:runtime.cwd}/data/" + folder


if __name__ == "__main__":

    for name, cls, extra in mfpbench.available():

        # For lcbench we need to basically edit the name to inculde the task
        if issubclass(cls, LCBenchBenchmark):
            filename = f"{name}_{extra['task_id']}"
        else:
            filename = name

        benchmark_name = filename

        # These two aren't a preset, we'll just skip
        if cls in [MFHartmann3Benchmark, MFHartmann6Benchmark]:
            continue

        path = HERE / f"{filename}.yaml"
        api = {"_target_": "mfpbench.get", "name": name, "seed": "${seed}"}

        # Add the path to where the experimental data is held if required
        datapath = datadir(cls)
        if datapath is not None:
            api["datadir"] = datapath

        # Add any extras
        if extra is not None:
            api.update(extra)

        config = {"name": benchmark_name, "api": api}

        with path.open("w") as f:
            yaml.dump(config, f, sort_keys=False)
