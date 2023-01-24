from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path
from dataclasses import dataclass
import json
import pandas as pd

HERE = Path(__file__).parent.absolute()
RESULTS_DIR = HERE / "results"

DEFAULT_BENCHMARKS = [
    "cifar100_wideresnet_2048",
    "imagenet_resnet_512",
    "jahs_cifar10",
    "jahs_colorectal_histology",
    "jahs_fashion_mnist",
    "lcbench-126026",
    "lcbench-167190",
    "lcbench-168330",
    "lcbench-168910",
    "lcbench-189906",
    "mfh3_good",
    "mfh3_terrible",
    "mfh6_good",
    "mfh6_terrible",
    "lm1b_transformer_2048",
    "translatewmt_xformer_64",
]


@dataclass
class BenchmarkResult:
    name: str
    means: list[float]
    stds: list[float]

    @classmethod
    def from_file(cls, path: Path) -> BenchmarkResult:
        with path.open("r") as f:
            results = json.load(f)

        return BenchmarkResult(
            name=path.stem,
            means=results["mean"],
            stds=results["std"],
        )

    def mean(self, z: float) -> float:
        assert 0 <= z <= 1
        index = int(z * len(self.means))
        return self.means[index]

    def std(self, z: float) -> float:
        assert 0 <= z <= 1
        index = int(z * len(self.means))
        return self.std[index]


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--zs",
        type=float,
        nargs="+",
        default=[0.1, 0.11, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    )
    parser.add_argument("--sort-by", type=float, default=0.11)
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--benchmarks", nargs="+", type=str, default=DEFAULT_BENCHMARKS)
    parser.add_argument("--no-mfh", action="store_true")
    parser.add_argument("--split-at", type=float, default=0.6)

    args = parser.parse_args()

    zs = args.zs
    results_dir = args.results_dir
    benchmarks = args.benchmarks

    assert all(0 <= z <= 1 for z in zs)
    assert results_dir.exists()

    results = [
        BenchmarkResult.from_file(f)
        for f in results_dir.iterdir()
        if any(f.name.startswith(b) for b in benchmarks)
    ]
    means = pd.DataFrame(
        {f"{result.name}_mean": [result.mean(z) for z in zs] for result in results},
        index=zs,
    ).transpose()
    c = f"sort-by-{args.sort_by}"
    means.insert(0, c, means[args.sort_by])

    if args.no_mfh:
        other_names = [
            f"{name}_mean" for name in args.benchmarks if not name.startswith("mfh")
        ]
        means = means.loc[other_names]

    upper = means[means[c] > args.split_at].sort_values(by=c, ascending=False)
    lower = means[means[c] <= args.split_at].sort_values(by=c, ascending=False)

    print("Upper")
    print(upper)

    print("Lower")
    print(lower)
