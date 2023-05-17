from __future__ import annotations

import datetime
from pathlib import Path
from typing import Any

from mfpbench.benchmark import Benchmark
from mfpbench.config import Config
from mfpbench.jahs import (
    JAHSBenchmark,
    JAHSCifar10,
    JAHSColorectalHistology,
    JAHSFashionMNIST,
)
from mfpbench.pd1 import (
    PD1Benchmark,
    PD1cifar100_wideresnet_2048,
    PD1imagenet_resnet_512,
    PD1lm1b_transformer_2048,
    PD1translatewmt_xformer_64,
    PD1uniref50_transformer_128,
)
from mfpbench.synthetic.hartmann import (
    MFHartmann3Benchmark,
    MFHartmann3BenchmarkBad,
    MFHartmann3BenchmarkGood,
    MFHartmann3BenchmarkModerate,
    MFHartmann3BenchmarkTerrible,
    MFHartmann6Benchmark,
    MFHartmann6BenchmarkBad,
    MFHartmann6BenchmarkGood,
    MFHartmann6BenchmarkModerate,
    MFHartmann6BenchmarkTerrible,
    MFHartmannBenchmark,
)
from mfpbench.yahpo import (
    IAMLglmnetBenchmark,
    IAMLrangerBenchmark,
    IAMLrpartBenchmark,
    IAMLSuperBenchmark,
    IAMLxgboostBenchmark,
    LCBenchBenchmark,
    NB301Benchmark,
    RBV2aknnBenchmark,
    RBV2glmnetBenchmark,
    RBV2rangerBenchmark,
    RBV2rpartBenchmark,
    RBV2SuperBenchmark,
    RBV2svmBenchmark,
    RBV2xgboostBenchmark,
    YAHPOBenchmark,
)

name = "mf-prior-bench"
package_name = "mfpbench"
author = "bobby1 and bobby2"
author_email = "**removed**"
description = "No description given"
url = "**removed**"
project_urls = {
    "Documentation": "**removed**",
    "Source Code": "**removed**",
}
copyright = f"Copyright {datetime.date.today().strftime('%Y')}, bobby1 and bobby2"
version = "0.0.1"

_mapping: dict[str, type[Benchmark]] = {
    # JAHS
    "jahs_cifar10": JAHSCifar10,
    "jahs_colorectal_histology": JAHSColorectalHistology,
    "jahs_fashion_mnist": JAHSFashionMNIST,
    # MFH
    "mfh3_terrible": MFHartmann3BenchmarkTerrible,
    "mfh3_bad": MFHartmann3BenchmarkBad,
    "mfh3_moderate": MFHartmann3BenchmarkModerate,
    "mfh3_good": MFHartmann3BenchmarkGood,
    "mfh6_terrible": MFHartmann6BenchmarkTerrible,
    "mfh6_bad": MFHartmann6BenchmarkBad,
    "mfh6_moderate": MFHartmann6BenchmarkModerate,
    "mfh6_good": MFHartmann6BenchmarkGood,
    # YAHPO
    "lcbench": LCBenchBenchmark,
    "nb301": NB301Benchmark,
    "rbv2_super": RBV2SuperBenchmark,
    "rbv2_aknn": RBV2aknnBenchmark,
    "rbv2_glmnet": RBV2glmnetBenchmark,
    "rbv2_ranger": RBV2rangerBenchmark,
    "rbv2_rpart": RBV2rpartBenchmark,
    "rbv2_svm": RBV2svmBenchmark,
    "rbv2_xgboost": RBV2xgboostBenchmark,
    "iaml_glmnet": IAMLglmnetBenchmark,
    "iaml_ranger": IAMLrangerBenchmark,
    "iaml_rpart": IAMLrpartBenchmark,
    "iaml_super": IAMLSuperBenchmark,
    "iaml_xgboost": IAMLxgboostBenchmark,
    # PD1
    "lm1b_transformer_2048": PD1lm1b_transformer_2048,
    "uniref50_transformer_128": PD1uniref50_transformer_128,
    "translatewmt_xformer_64": PD1translatewmt_xformer_64,
    "cifar100_wideresnet_2048": PD1cifar100_wideresnet_2048,
    "imagenet_resnet_512": PD1imagenet_resnet_512,
}


def get(
    name: str,
    seed: int | None = None,
    prior: str | Path | Config | None = None,
    preload: bool = False,
    **kwargs: Any,
) -> Benchmark:
    """Get a benchmark.

    ```python
    import mfpbench
    b = mfpbench.get("jahs_cifar_10", datadir=..., seed=1)
    cs = b.space()

    sample = cs.sample_configuration()

    result = b.query(sample, at=42)
    print(result)
    ```

    If using something with a task:

    ```python
    b = mfpbench.get("lcbench", task="3945", datadir=..., seed=1)
    ```

    Parameters
    ----------
    name : str
        The name of the benchmark

    seed: int | None = None
        The seed to use

    prior: str | Path | Config | None = None
        The prior to use for the benchmark.
        * str -
            If it ends in {.json} or {.yaml, .yml}, it will convert it to a path and use
            it as if it is a path to a config. Otherwise, it is treated as a preset
        * Path - path to a file
        * Config - A Config object
        * None - Use the default if available

    preload: bool = False
        Whether to preload the benchmark data in

    **kwargs: Any
        Extra arguments, optional or required for other benchmarks

        YAHPOBenchmark
        --------------
        datadir: str | Path | None = None
            Path to where the benchmark should look for data if it does. Defaults to
            "./data/yahpo-gym-data"

        task_id : str | None = None
            A task id if the yahpo bench requires one

        JAHSBenchmark
        -------------
        datadir: str | Path | None = None
            Path to where the benchmark should look for data if it does. Defaults to
            "./data/jahs-bench-data"

        MFHartmann3Benchmark
        --------------------
        bias: float
            Amount of bias, realized as a flattening of the objective.

        noise: float
            Amount of noise, decreasing linearly (in st.dev.) with fidelity.
    """
    b = _mapping.get(name, None)
    bench: Benchmark

    if b is None:
        raise ValueError(f"{name} is not a benchmark in {list(_mapping.keys())}")

    if isinstance(prior, str):
        if any(prior.endswith(suffix) for suffix in [".json", ".yaml", ".yml"]):
            prior = Path(prior)

    bench = b(seed=seed, prior=prior, **kwargs)

    if preload:
        bench.load()

    return bench


__all__ = [
    "name",
    "package_name",
    "author",
    "author_email",
    "description",
    "url",
    "project_urls",
    "copyright",
    "version",
    "get",
    "JAHSBenchmark",
    "YAHPOBenchmark",
    "PD1Benchmark",
    "MFHartmannBenchmark",
    "MFHartmann3Benchmark",
    "MFHartmann6Benchmark",
]
