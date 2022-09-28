"""Generates all possible benchmark configs from mfpbench."""
from __future__ import annotations

from itertools import product
from pathlib import Path
from typing import Any, Iterator

import mfpbench
import yaml
from mfpbench import JAHSBenchmark, MFHartmannBenchmark, PD1Benchmark, YAHPOBenchmark

HERE = Path(__file__).parent.resolve()

EXCLUDE = ["rbv2", "iaml"]
LCBENCH_TASKS = ["189862", "189862", "189866"]

# Whether to generate configs for condtional spaces
CONDITONAL_HP_SPACES = False

# @carl, change them here as you need
HARTMANN_NOISY_PRIOR_VALUES = [0.250]
AVAILABLE_PRIORS = ["good", "medium", "bad"]


def hartmann_configs() -> Iterator[tuple[str, dict[str, Any]]]:
    names = [
        f"mfh{i}_{corr}"
        for i, corr in product([3, 6], ["terrible", "bad", "moderate", "good"])
    ]

    for name, prior in product(names, AVAILABLE_PRIORS + ["perfect"]):
        config_name = f"{name}_prior-{prior}"
        api = {"name": name, "prior": prior}

        yield config_name, api

        # We also give a noisy prior version for each
        for noise_scale in HARTMANN_NOISY_PRIOR_VALUES:
            # TODO: This is a last minute fix
            if noise_scale == 0.250:
                noise_scale_str = "0.125"
            else:
                noise_scale_str = str(noise_scale)
            config_name = f"{config_name}-noisy{noise_scale_str}"
            yield config_name, {
                **api,
                "noisy_prior": True,
                "prior_noise_scale": noise_scale,
            }


def pd1_configs() -> Iterator[tuple[str, dict[str, Any]]]:
    datadir = "pd1-data"
    names = [
        "lm1b_transformer_2048",
        "uniref50_transformer_128",
        "translatewmt_xformer_64",
    ]

    for name, prior in product(names, AVAILABLE_PRIORS):
        config_name = f"{name}_prior-{prior}"
        api = {
            "name": name,
            "prior": prior,
            "datadir": "${hydra:runtime.cwd}/data/" + datadir,
        }

        yield config_name, api


def yahpo_configs() -> Iterator[tuple[str, dict[str, Any]]]:
    datadir = "yahpo-gym-data"

    rbv2_names = [
        f"rbv2_{x}"
        for x in ("super", "aknn", "glmnet", "ranger", "rpart", "svm", "xgboost")
    ]
    iaml_names = [f"rbv2_{x}" for x in ("super", "glmnet", "ranger", "rpart", "xgboost")]
    names = ["lcbench", "nb301"] + rbv2_names + iaml_names

    for name, prior in product(names, AVAILABLE_PRIORS):
        bench = mfpbench._mapping[name]
        assert issubclass(bench, YAHPOBenchmark)

        # Skip conditional spaces if we must
        if bench.has_conditionals and not CONDITONAL_HP_SPACES:
            continue

        api = {
            "name": name,
            "prior": prior,
            "datadir": "${hydra:runtime.cwd}/data/" + datadir,
        }
        if bench.instances is None:
            config_name = f"{name}_prior-{prior}"
            yield config_name, api
        else:
            for task_id in bench.instances:
                config_name = f"{name}-{task_id}_prior-{prior}"
                yield config_name, {**api, "task_id": task_id}


def jahs_configs() -> Iterator[tuple[str, dict[str, Any]]]:
    datadir = "jahs-bench-data"

    names = ["jahs_cifar10", "jahs_colorectal_histology", "jahs_fashion_mnist"]

    for name, prior in product(names, AVAILABLE_PRIORS):
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
        PD1Benchmark: pd1_configs,
    }
    generators = [generator for cls, generator in mapping.items()]

    for generator in generators:
        for config_name, api in generator():
            if any(config_name.startswith(e) for e in EXCLUDE):
                continue

            if config_name.startswith("lcbench"):
                if not any(i in config_name for i in LCBENCH_TASKS):
                    continue

            # Put in defaults for each config
            api.update({"_target_": "mfpbench.get", "seed": "${seed}", "preload": True})

            # Create the config and filename
            config: dict[str, Any] = {"name": config_name, "api": api}
            filename = f"{config_name}.yaml"
            path = HERE / filename

            # Get the scores for the prior
            # Seed isnt need as prior is deterministic
            kwargs = {k: v for k, v in config["api"].items() if k not in ["datadir", "_target_", "seed"]}
            b = mfpbench.get(**kwargs)
            if b.prior:
                results = b.trajectory(b.prior)
                highest_fidelity_error = results[-1].error
                lowest_error = min(results, key=lambda r: r.error).error

                config["prior_highest_fidelity_error"] = float(highest_fidelity_error)
                config["prior_lowest_error"] = float(lowest_error)

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
