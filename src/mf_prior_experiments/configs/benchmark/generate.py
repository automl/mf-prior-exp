"""Generates all possible benchmark configs from mfpbench."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Iterator

import mfpbench
import yaml
from mfpbench import (
    Benchmark,
    JAHSBenchmark,
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


def configs() -> Iterator[tuple[Path, dict[str, Any]]]:

    # Things to exclude from filenames from `extra`
    filename_ignores = [
        # Part of the Hartmann benchmarks, not needed, too noisy to see
        "bias",
        "noise",
    ]

    for name, cls, prior, extra in mfpbench.available():

        if "jahs" in name:
            print(name, prior, extra)
        filename_items = None
        if extra is not None:
            filename_items = {k: v for k, v in extra.items() if k not in filename_ignores}
            if len(filename_items) == 0:
                filename_items = None

        if filename_items:
            item_str = "_".join([f"{k}-{v}" for k, v in filename_items.items()])
            filename = f"{name}_{item_str}"
        else:
            filename = name

        if prior is not None:
            filename = f"{filename}_prior-{prior}"

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

        # Add the prior if it's there
        if prior is not None:
            api["prior"] = prior

        # Add any extras
        if extra is not None:
            api.update(extra)

        config = {"name": benchmark_name, "api": api}
        yield path, config


if __name__ == "__main__":
    for path, config in configs():
        with path.open("w") as f:
            yaml.dump(config, f, sort_keys=False)
