from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path
from dataclasses import dataclass
import json

HERE = Path(__file__).parent.absolute()
RESULTS_DIR = HERE / "results"


@dataclass
class BenchmarkResult:
    task_id: int
    means: list[float]
    stds: list[float]

    @classmethod
    def from_file(cls, path: Path) -> BenchmarkResult:
        task_id = int(path.stem.split("-")[-1])
        with path.open("r") as f:
            results = json.load(f)

        return BenchmarkResult(
            task_id=task_id,
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
    parser = ArgumentParser(
        description=(
            "Choose lcbench tasks based on `quantiles`\n"
            " and ranked at `z`% along it's correlation curve"
        )
    )
    parser.add_argument("-z", type=float, required=True)
    parser.add_argument("--quantiles", nargs="+", type=float, required=True)
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)

    args = parser.parse_args()

    z = args.z
    quantiles = args.quantiles
    results_dir = args.results_dir

    assert 0 <= z <= 1
    assert all(0 <= q <= 1 for q in quantiles)
    assert results_dir.exists()

    results = [
        BenchmarkResult.from_file(f)
        for f in results_dir.iterdir()
        if f.name.startswith("lcbench")
    ]
    n_results = len(results)

    # Since correlation 1 is good and 0 is bad, we sort with worse at the start
    # such that .95 will be near the end of the array and mean .95 of the others
    # were below
    sorted_results = sorted(results, key=lambda r: r.mean(z))
    indices = [int(q * (n_results - 1)) for q in quantiles]
    selected_results = [sorted_results[i] for i in indices]

    for result in selected_results:
        print(result.task_id)
