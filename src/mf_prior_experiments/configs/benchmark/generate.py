"""Generates all possible benchmark configs from mfpbench.

NB:
    Search this file for TODO to find editing points
"""
from __future__ import annotations

from itertools import product
from pathlib import Path
from typing import Any, Iterator, Callable

import mfpbench
import yaml
from mfpbench import MFHartmannBenchmark, YAHPOBenchmark

HERE = Path(__file__).parent.resolve()

CONFIGSPACE_SEED = 133_077

JAHS_BENCHMARKS = ["jahs_cifar10", "jahs_colorectal_histology", "jahs_fashion_mnist"]
PD1_DATASETS = [
    "lm1b_transformer_2048",
    #"uniref50_transformer_128",
    "translatewmt_xformer_64",
    "imagenet_resnet_512",
    "cifar100_wideresnet_2048",
]
LCBENCH_TASKS = ["126026", "167190", "168910", "168330", "189906"]
HARTMANN_BENCHMARKS = [
    f"mfh{i}_{corr}" for i, corr in product([3, 6], ["terrible", "good"])
]

# TODO: Edit as required
PRIORS_TO_DO = ["medium", "good", "at25", "bad"]
MEDIUM_PERTURB_STRENGTH = 0.25

def hartmann_configs() -> Iterator[tuple[str, dict[str, Any]]]:

    for name, prior in product(HARTMANN_BENCHMARKS, PRIORS_TO_DO):
        api: dict = {
            "name": name,
            "prior": prior,
        }

        if prior == "good":
            api.update({"noisy_prior": True, "prior_noise_scale": 0.250 })

        if prior == "medium":
            api.update({"perturb_prior": MEDIUM_PERTURB_STRENGTH})

        config_name = f"{name}_prior-{prior}"
        yield config_name, api


def pd1_configs() -> Iterator[tuple[str, dict[str, Any]]]:
    datadir = "pd1-data"
    for name, prior in product(PD1_DATASETS, PRIORS_TO_DO):
        config_name = f"{name}_prior-{prior}"
        api: dict = {
            "name": name,
            "prior": prior,
            "datadir": "${hydra:runtime.cwd}/data/" + datadir,
        }

        if prior == "medium":
            api.update({"perturb_prior": MEDIUM_PERTURB_STRENGTH})

        yield config_name, api


def lcbench_configs() -> Iterator[tuple[str, dict[str, Any]]]:
    datadir = "yahpo-gym-data"

    for task_id, prior in product(LCBENCH_TASKS, PRIORS_TO_DO):
        bench = mfpbench._mapping["lcbench"]

        assert issubclass(bench, YAHPOBenchmark)

        api: dict = {
            "name": "lcbench",
            "prior": prior,
            "datadir": "${hydra:runtime.cwd}/data/" + datadir,
            "task_id": task_id,
        }

        if prior == "medium":
            api.update({"perturb_prior": MEDIUM_PERTURB_STRENGTH})

        config_name = f"lcbench-{task_id}_prior-{prior}"
        yield config_name, api


def jahs_configs() -> Iterator[tuple[str, dict[str, Any]]]:
    datadir = "jahs-bench-data"

    for name, prior in product(JAHS_BENCHMARKS, PRIORS_TO_DO):
        config_name = f"{name}_prior-{prior}"
        api: dict = {
            "name": name,
            "datadir": "${hydra:runtime.cwd}/data/" + datadir,
            "prior": prior,
        }

        if prior == "medium":
            api.update({"perturb_prior": MEDIUM_PERTURB_STRENGTH})

        yield config_name, api


def configs() -> Iterator[tuple[Path, dict[str, Any]]]:
    """Generate all configs we might care about for the benchmark."""
    # TODO: Edit as required
    generators: list[Callable[[], Iterator[tuple[str, dict[str, Any]]]]] = [
        #lcbench_configs,
        #jahs_configs,
        #hartmann_configs,
        pd1_configs,
    ]
    for generator in generators:

        for config_name, api in generator():
            print(f"{config_name}")
            # Put in defaults for each config
            api.update(
                {
                    "_target_": "mfpbench.get",
                    "seed": "${seed}",
                    "preload": True,
                }
            )

            # Create the config and filename
            config: dict[str, Any] = {
                "name": config_name,
                "api": api,
            }
            filename = f"{config_name}.yaml"
            path = HERE / filename

            # Get the scores for the prior
            kwargs = {
                k: v
                for k, v in config["api"].items()
                # We pass a real seed and remove perturbations
                if k not in ["datadir", "_target_", "seed", "perturb_prior"]
            }

            b = mfpbench.get(seed=CONFIGSPACE_SEED, **kwargs)
            if b.prior:
                print(f" - {b.prior}")
                results = b.trajectory(b.prior)
                highest_fidelity_error = results[-1].error
                lowest_error = min(results, key=lambda r: r.error).error

                config["prior_highest_fidelity_error"] = float(highest_fidelity_error)
                config["prior_lowest_error"] = float(lowest_error)
                print(f"   - error {highest_fidelity_error}")
                del b

            # We also give the best score of RS for 10, 25, 100
            # We remove information about the priors to keep it random
            # We also reload the benchmark at each iteration just because
            # sampling 100 and taking the first 25 is not the same
            # as sampling just 25
            for key in ["prior", "noisy_prior", "prior_noise_scale", "perturb_prior"]:
                kwargs.pop(key, None)

            for i in (10, 25, 50, 90, 100):
                b = mfpbench.get(seed=CONFIGSPACE_SEED, **kwargs)

                print(f"  - sampling {i}")
                configs = b.sample(i)

                print(f"  - querying {i}")
                results = [b.query(c) for c in configs]
                best = min(results, key=lambda r: r.error)

                config[f"best_{i}_error"] = float(best.error)
                print(f"  - config: {config}")

            if isinstance(b, MFHartmannBenchmark):
                result = b.query(b.optimum)
                config["optimum"] = float(result.error)

            yield path, config


if __name__ == "__main__":
    for path, config in configs():
        with path.open("w") as f:
            yaml.dump(config, f, sort_keys=False)
