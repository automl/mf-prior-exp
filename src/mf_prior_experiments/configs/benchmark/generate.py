"""Generates all possible benchmark configs from mfpbench."""
from __future__ import annotations

from itertools import product
from pathlib import Path
from typing import Any, Iterator

import mfpbench
import yaml
from mfpbench import JAHSBenchmark, MFHartmannBenchmark, YAHPOBenchmark

HERE = Path(__file__).parent.resolve()


# Don't generate benchmark yamls for these
DONT_GENERATE_CLASSES = (YAHPOBenchmark,)

# Whether to generate configs for condtional spaces
CONDITONAL_HP_SPACES = False

# @carl, change them here as you need
HARTMANN_NOISY_PRIOR_VALUES = [0.125]


def hartmann_configs() -> Iterator[tuple[str, dict[str, Any]]]:
    names = [
        f"mfh{i}_{corr}"
        for i, corr in product([3, 6], ["terrible", "bad", "moderate", "good"])
    ]

    for name in names:
        bench = mfpbench._mapping[name]
        assert issubclass(bench, MFHartmannBenchmark)

        for prior in bench.available_priors:
            config_name = f"{name}_prior-{prior}"
            api = {"name": name, "prior": prior}

            yield config_name, api

            # We also give a noisy prior version for each
            for noise_scale in HARTMANN_NOISY_PRIOR_VALUES:
                config_name = f"{config_name}-noisy{str(noise_scale)}"
                yield config_name, {
                    **api,
                    "noisy_prior": True,
                    "prior_noise_scale": noise_scale,
                }


def yahpo_configs() -> Iterator[tuple[str, dict[str, Any]]]:
    datadir = "yahpo-gym-data"

    rbv2_names = [
        f"rbv2_{x}"
        for x in ("super", "aknn", "glmnet", "ranger", "rpart", "svm", "xgboost")
    ]
    iaml_names = [f"rbv2_{x}" for x in ("super", "glmnet", "ranger", "rpart", "xgboost")]
    names = ["lcbench", "nb301"] + rbv2_names + iaml_names

    for name in names:
        bench = mfpbench._mapping[name]
        assert issubclass(bench, YAHPOBenchmark)

        # Skip conditional spaces if we must
        if bench.has_conditionals and not CONDITONAL_HP_SPACES:
            continue

        config_name = f"{name}"
        api = {
            "name": name,
            "datadir": "${hydra:runtime.cwd}/data/" + datadir,
        }
        if bench.instances is None:
            yield config_name, api
        else:
            for task_id in bench.instances:
                yield config_name, {**api, "task_id": task_id}


def jahs_configs() -> Iterator[tuple[str, dict[str, Any]]]:
    datadir = "jahs-bench-data"

    names = ["jahs_cifar10", "jahs_colorectal_histology", "jahs_fashion_mnist"]

    for name in names:
        bench = mfpbench._mapping[name]
        assert issubclass(bench, JAHSBenchmark)

        for prior in bench.available_priors:
            config_name = f"{name}_prior-{prior}"
            api = {
                "name": name,
                "datadir": "${hydra:runtime.cwd}/data/" + datadir,
                "prior": prior,
            }
            yield config_name, api


def configs() -> Iterator[tuple[Path, dict[str, Any]]]:
    """Generate all configs we might care about for the benchmark."""
    mapping = {
        YAHPOBenchmark: yahpo_configs,
        JAHSBenchmark: jahs_configs,
        MFHartmannBenchmark: hartmann_configs,
    }
    generators = [
        generator
        for cls, generator in mapping.items()
        if not issubclass(cls, DONT_GENERATE_CLASSES)
    ]

    for generator in generators:
        for config_name, api in generator():
            # Put in defaults for each config
            api.update({"_target_": "mfpbench.get", "seed": "${seed}", "preload": True})

            # Create the config and filename
            config = {"name": config_name, "api": api}
            filename = f"{config_name}.yaml"
            path = HERE / filename

            yield path, config


if __name__ == "__main__":
    for path, config in configs():
        with path.open("w") as f:
            yaml.dump(config, f, sort_keys=False)
