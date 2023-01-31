from __future__ import annotations

import operator
from dataclasses import dataclass, field, replace
from itertools import accumulate, groupby, product, starmap
from multiprocessing import Pool
from pathlib import Path
from typing import Any, Callable, Sequence, overload

import mfpbench
import pandas as pd
import yaml  # type: ignore
from more_itertools import all_equal, all_unique, first, pairwise


@dataclass
class Config:
    # 0_0, 1_0, 2_0, 0_1, ..., 50_3
    id: int
    bracket: int
    params: dict[str, Any]

    def as_tuple(self) -> tuple[int, int]:
        return (self.id, self.bracket)

    def __str__(self) -> str:
        return f"{self.id}_{self.bracket}"

    def is_direct_continuation(self, of: Config) -> bool:
        return self.id == of.id and of.bracket == self.bracket - 1

    @classmethod
    def parse(cls, dirname: str, config: dict) -> Config:
        id_, bracket = map(int, dirname.split("_"))
        return Config(id_, bracket, params=config)


@dataclass
class Result:
    config: Config = field(repr=False)
    loss: float
    cost: float
    val_score: float
    test_score: float
    fidelity: int
    start_time: float
    end_time: float
    max_fidelity_loss: float
    max_fidelity_cost: float
    start_time_since_global_start: float | None = None
    end_time_since_global_start: float | None = None
    process_id: int | None = None

    @classmethod
    def from_dir(cls, config_dir: Path) -> Result:
        config_yaml = config_dir / "config.yaml"
        result_yaml = config_dir / "result.yaml"
        # metadata_yaml = config_dir / "metadata.yaml"

        with config_yaml.open("r") as f:
            config = yaml.safe_load(f)

        with result_yaml.open("r") as f:
            result = yaml.safe_load(f)

        info = result["info"]
        return cls(
            config=Config.parse(config_dir.name, config),
            loss=result["loss"],
            cost=result["cost"],
            val_score=info["val_score"],
            test_score=info["test_score"],
            fidelity=info["fidelity"],
            start_time=info["start_time"],
            end_time=info["end_time"],
            max_fidelity_loss=info["max_fidelity_loss"],
            max_fidelity_cost=info["max_fidelity_cost"],
            process_id=info.get("process_id"),
        )

    def continue_from(self, other: Result) -> Result:
        """Continue based on the results from a previous evaluation of the same config."""
        assert self.is_continuation(of=other)
        changes = {
            "fidelity": self.fidelity - other.fidelity,
            "cost": self.cost - other.cost,
        }
        return self.mutate(**changes)

    def is_continuation(self, of: Result) -> bool:
        return self.config.is_direct_continuation(of.config)

    def mutate(self, **kwargs: Any) -> Result:
        return replace(self, **kwargs)


@dataclass
class Algorithm:
    name: str

    def __hash__(self) -> int:
        return hash(self.name)


@dataclass
class Trace(Sequence[Result]):
    experiment_group: str
    algorithm: Algorithm
    benchmark: Benchmark
    seed: int
    results: list[Result] = field(repr=False)

    @classmethod
    def load(
        cls,
        base_path: Path,
        experiment_group: str,
        algorithm: Algorithm,
        benchmark: Benchmark,
        seed: int,
    ) -> Trace:
        results_dir = (
            base_path
            / experiment_group
            / f"benchmark={benchmark}"
            / f"algorithm={algorithm}"
            / f"seed={seed}"
            / "neps_root_directory"
            / "results"
        )

        if not results_dir.exists():
            raise ValueError(f"Results dir {results_dir} does not exist")

        results = [Result.from_dir(config_dir) for config_dir in results_dir.iterdir()]
        if not all_unique(results, key=lambda r: (r.config.id, r.config.bracket)):
            msg = f"Found duplicate results for {benchmark}-{algorithm}-{seed}!"
            raise RuntimeError(msg)

        global_start = min(r.start_time for r in results)
        results = [
            result.mutate(
                start_time_since_global_start=result.start_time - global_start,
                end_time_since_global_start=result.end_time - global_start,
            )
            for result in results
        ]

        results = sorted(results, key=lambda r: r.end_time)
        return cls(
            experiment_group=experiment_group,
            algorithm=algorithm,
            benchmark=benchmark,
            seed=seed,
            results=results,
        )

    @classmethod
    def seeds_available(
        cls,
        base_path: Path,
        experiment_group: str,
        algorithm: Algorithm,
        benchmark: Benchmark,
    ) -> list[int]:
        algo_dir = (
            base_path
            / experiment_group
            / f"benchmark={benchmark}"
            / f"algorithm={algorithm}"
        )
        return [
            int(d.name.split("=")[1])
            for d in algo_dir.iterdir()
            if d.is_dir() and "seed" in d.name
        ]

    @classmethod
    def load_all(
        cls,
        base_path: Path,
        experiment_group: str,
        algorithms: list[Algorithm],
        benchmarks: list[Benchmark],
        parallel: bool = False,
    ) -> list[Trace]:
        seeds_present = {
            (benchmark, algorithm): sorted(
                cls.seeds_available(
                    base_path=base_path,
                    experiment_group=experiment_group,
                    algorithm=algorithm,
                    benchmark=benchmark,
                )
            )
            for benchmark, algorithm in product(benchmarks, algorithms)
        }
        if not all_equal(seeds_present.values()):
            msg = f"Not all (benchmark, algorithm) have the same seeds present!\n{seeds_present}"
            raise RuntimeError(msg)

        seeds: list[int] = first(seeds_present.values())
        trace_itr = product(
            [base_path], [experiment_group], algorithms, benchmarks, seeds
        )

        if parallel:
            with Pool() as pool:
                traces = pool.starmap(Trace.load, trace_itr)
        else:
            traces = list(starmap(Trace.load, trace_itr))

        return traces

    @overload
    def __getitem__(self, key: int) -> Result:
        ...

    @overload
    def __getitem__(self, key: slice) -> list[Result]:
        ...

    def __getitem__(self, key: int | slice) -> Result | list[Result]:
        return self.results[key]

    def __len__(self) -> int:
        return len(self.results)

    @property
    def df(self) -> pd.DataFrame:
        df = pd.DataFrame(
            data=[
                {
                    "loss": result.loss,
                    "cost": result.cost,
                    "val_score": result.val_score,
                    "test_score": result.test_score,
                    "fidelity": result.fidelity,
                    "start_time": result.start_time,
                    "end_time": result.end_time,
                    "max_fidelity_loss": result.max_fidelity_loss,
                    "max_fidelity_cost": result.max_fidelity_cost,
                    "config_id": str(result.config.id),
                    "process_id": result.process_id,
                }
                for result in self.results
            ]
        )
        df = df.set_index("config_id", drop=True)
        assert df is not None
        return df

    def with_continuations(self) -> Trace:
        """Add results for continuations of configs that were evaluated before."""
        # Group the results by the config id and then sort them by bracket
        # {
        #   0: [0_0, 0_1, 0_2]
        #   1: [1_0]
        #   2: [2_0, 2_1],
        # }
        results = {
            config_id: sorted(results, key=lambda r: r.config.bracket)
            for config_id, results in groupby(self.results, key=lambda r: r.config.id)
        }

        continuations = []
        for config_results in results.values():
            assert len(config_results) == max(r.config.bracket for r in config_results)

            # Put the lowest bracket entry into the continued results,
            # it can't have continued from anything
            continuations.append(config_results[0])

            if len(config_results) == 1:
                continue

            # We have more than one evaluation for this config (assumingly at a higher bracket)
            for lower_bracket, higher_bracket in pairwise(config_results):
                continued_result = higher_bracket.continue_from(lower_bracket)
                continuations.append(continued_result)

        sorted_continuations = sorted(continuations, key=lambda r: r.end_time)
        return replace(self, results=sorted_continuations)

    def with_cumulative_fidelity(self) -> Trace:
        """This only really makes sense for traces generated by single workers"""
        assert all_equal(r.process_id for r in self.results)
        results = sorted(self.results, key=lambda r: r.end_time)
        cumulated_fidelities = accumulate([r.fidelity for r in results])
        cumulated_results = [
            r.mutate(fidelity=f) for r, f in zip(results, cumulated_fidelities)
        ]
        return replace(self, results=cumulated_results)

    def incumbent_trace(
        self,
        xaxis: str | Callable[[Result], float] = lambda r: r.fidelity,
        yaxis: str | Callable[[Result], float] = lambda r: r.max_fidelity_loss,
        *,
        op: Callable[[float, float], bool] = operator.lt,
    ) -> Trace:
        """Return a trace with only the incumbent results."""
        if isinstance(xaxis, str):
            xaxis = lambda r: getattr(r, xaxis)  # type: ignore
        if isinstance(yaxis, str):
            yaxis = lambda r: getattr(r, yaxis)  # type: ignore

        results: list[Result] = sorted(self.results, key=xaxis)
        incumbent = results[0]

        incumbents = [incumbent]
        for result in results[1:]:
            # If the new result is better than the incumbent, replace the incumbent
            if op(yaxis(result), yaxis(incumbent)):
                incumbent = result
                incumbents.append(incumbent)

        return replace(self, results=incumbents)

    def series(self, index: str, values: str) -> pd.Series:
        vals = self.df[values]
        indices = self.df[index]
        name = f"{self.benchmark}-{self.algorithm}-{self.seed}"
        series = pd.Series(vals, index=indices, name=name)
        return series

    @classmethod
    def combine(
        cls,
        traces: list[Trace],
        *,
        xaxis: str = "fidelity",  # end_time_since_global_start
        yaxis: str = "loss",  # max_fidelity_loss
        nan_fill: bool = True,
    ) -> pd.DataFrame:
        """Combine multiple traces into a single dataframe."""
        trace_series = [trace.series(index=xaxis, values=yaxis) for trace in traces]
        df = pd.concat(trace_series, axis=1).sort_index(ascending=True)  # column wise
        assert isinstance(df, pd.DataFrame)

        if nan_fill:
            df = df.fillna(method="backfill", axis="rows")
            df = df.fillna(method="ffill", axis="rows")

        assert isinstance(df, pd.DataFrame)
        return df


@dataclass
class Benchmark:
    name: str
    basename: str
    prior: str
    task_id: str | None
    best_10_error: float
    best_25_error: float
    best_50_error: float
    best_90_error: float
    best_100_error: float
    prior_error: float
    optimum: float | None  # Only for mfh
    epsilon: float | None  # Only for some priors
    _config_path: Path
    _config: dict
    _benchmark: mfpbench.Benchmark | None  # Lazy loaded

    @classmethod
    def from_name(cls, name: str, config_dir: Path) -> Benchmark:
        expected_path = config_dir / "configs" / "benchmark" / f"{name}.yaml"
        if not expected_path.exists():
            raise ValueError(f"Expected benchmark path {expected_path} to exist.")

        with expected_path.open("r") as f:
            config = yaml.safe_load(f)

        return cls(
            name=name,
            basename=config["api"]["name"],
            prior=config["api"]["prior"],
            epsilon=config["api"].get("epsilon"),
            task_id=config["api"].get("task_id"),
            optimum=config["api"].get("optimum"),
            prior_error=config["api"]["prior_highest_fidelity_error"],
            best_10_error=config["api"]["best_10_error"],
            best_25_error=config["api"]["best_25_error"],
            best_50_error=config["api"]["best_50_error"],
            best_90_error=config["api"]["best_90_error"],
            best_100_error=config["api"]["best_100_error"],
            _config_path=expected_path,
            _config=config,
            _benchmark=None,
        )

    @property
    def benchmark(self) -> mfpbench.Benchmark:
        if self._benchmark is None:
            if self.task_id is not None:
                self._benchmark = mfpbench.get(self.name, task_id=self.task_id)
            else:
                self._benchmark = mfpbench.get(self.name)

        return self._benchmark

    @property
    def max_fidelity(self) -> int | float:
        return self.benchmark.end

    def __hash__(self) -> int:
        return hash(self.name)

    def __str__(self) -> str:
        return self.name

    def __repr__(self) -> str:
        return str(self)
