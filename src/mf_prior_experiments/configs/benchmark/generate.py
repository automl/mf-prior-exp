"""Generates all possible benchmark configs from mfpbench."""
from __future__ import annotations

from itertools import product
from pathlib import Path
from typing import Any, Iterator

import mfpbench
import yaml
from mfpbench import MFHartmannBenchmark, YAHPOBenchmark

HERE = Path(__file__).parent.resolve()

CONFIGSPACE_SEED = 133_077

JAHS_BENCHMARKS = ["jahs_cifar10", "jahs_colorectal_histology", "jahs_fashion_mnist"]
JAHS_PRIORS = ["bad", "good"]

PD1_DATASETS = [
    "lm1b_transformer_2048",
    "uniref50_transformer_128",
    "translatewmt_xformer_64",
    "imagenet_resnet_512",
    "cifar100_wideresnet_2048",
]
PD1_PRIORS = ["bad", "good"]

LCBENCH_TASKS = ["126026", "167190", "168910", "168330", "189906"]
LCBENCH_PRIORS = ["bad", "good"]

HARTMANN_BENCHMARKS = [
    f"mfh{i}_{corr}" for i, corr in product([3, 6], ["terrible", "good"])
]
HARTMANN_PRIORS = ["bad", "perfect-noisy0.25"]


def hartmann_configs() -> Iterator[tuple[str, dict[str, Any]]]:

    for name, prior in product(HARTMANN_BENCHMARKS, HARTMANN_PRIORS):
        api: dict = {
            "name": name,
            "prior": prior,
        }

        config_name = f"{name}_prior-{prior}"
        yield config_name, api


def pd1_configs() -> Iterator[tuple[str, dict[str, Any]]]:
    datadir = "pd1-data"
    for name, prior in product(PD1_DATASETS, PD1_PRIORS):
        config_name = f"{name}_prior-{prior}"
        api = {
            "name": name,
            "prior": prior,
            "datadir": "${hydra:runtime.cwd}/data/" + datadir,
        }

        yield config_name, api


def lcbench_configs() -> Iterator[tuple[str, dict[str, Any]]]:
    datadir = "yahpo-gym-data"

    for task_id, prior in product(LCBENCH_TASKS, LCBENCH_PRIORS):
        bench = mfpbench._mapping["lcbench"]

        assert issubclass(bench, YAHPOBenchmark)

        api = {
            "name": "lcbench",
            "prior": prior,
            "datadir": "${hydra:runtime.cwd}/data/" + datadir,
            "task_id": task_id,
        }
        config_name = f"lcbench-{task_id}_prior-{prior}"
        yield config_name, api


def jahs_configs() -> Iterator[tuple[str, dict[str, Any]]]:
    datadir = "jahs-bench-data"

    for name, prior in product(JAHS_BENCHMARKS, JAHS_PRIORS):
        config_name = f"{name}_prior-{prior}"
        api = {
            "name": name,
            "datadir": "${hydra:runtime.cwd}/data/" + datadir,
            "prior": prior,
        }
        yield config_name, api


def configs() -> Iterator[tuple[Path, dict[str, Any]]]:
    """Generate all configs we might care about for the benchmark."""
    generators = [
        lcbench_configs,
        jahs_configs,
        hartmann_configs,
        pd1_configs
    ]
    for generator in generators:

        for config_name, api in generator():
            # Put in defaults for each config
            api.update({"_target_": "mfpbench.get", "seed": "${seed}", "preload": True})

            # Create the config and filename
            config: dict[str, Any] = {"name": config_name, "api": api}
            filename = f"{config_name}.yaml"
            path = HERE / filename

            # Get the scores for the prior
            kwargs = {
                k: v
                for k, v in config["api"].items()
                if k not in ["datadir", "_target_", "seed"]  # We pass a real seed
            }

            b = mfpbench.get(seed=CONFIGSPACE_SEED, **kwargs)
            if b.prior:
                results = b.trajectory(b.prior)
                highest_fidelity_error = results[-1].error
                lowest_error = min(results, key=lambda r: r.error).error

                config["prior_highest_fidelity_error"] = float(highest_fidelity_error)
                config["prior_lowest_error"] = float(lowest_error)

            # We also give the best score of RS for 10, 25, 100
            # We remove information about the priors to keep it random
            del kwargs["prior"]

            b = mfpbench.get(seed=CONFIGSPACE_SEED, **kwargs)
            configs = b.sample(100)
            results = [b.query(c) for c in configs]

            for i in (10, 25, 50, 90, 100):
                best = min(results[:i], key=lambda r: r.error)
                config[f"best_{i}_error"] = float(best.error)

            if isinstance(b, MFHartmannBenchmark):
                optimum = b.Config.from_dict(
                    {f"X_{i}": x for i, x in enumerate(b.Generator.optimum)}
                )
                result = b.query(optimum)
                config["optimum"] = float(result.error)

            yield path, config


if __name__ == "__main__":
    for path, config in configs():
        with path.open("w") as f:
            yaml.dump(config, f, sort_keys=False)
