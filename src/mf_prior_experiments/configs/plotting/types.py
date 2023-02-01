from __future__ import annotations

import operator
from dataclasses import dataclass, field, replace
from itertools import accumulate, chain, groupby, starmap
from multiprocessing.pool import Pool
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, Sequence, overload, Iterable

import mfpbench
import pandas as pd
import yaml  # type: ignore
from more_itertools import all_equal, pairwise
from typing_extensions import Literal

# NOTE: These need to be standalone functions for
# it to work with multiprocessing
def _with_continuations(t: Trace) -> Trace:
    return t.with_continuations()


def _with_cumulative_fidelity(t: Trace) -> Trace:
    return t.with_cumulative_fidelity()


def _incumbent_trace(t: Trace, xaxis: str, yaxis: str) -> Trace:
    return t.incumbent_trace(xaxis=xaxis, yaxis=yaxis)


def _rescale(t: Trace, xaxis: str, c: float) -> Trace:
    return t.rescale(xaxis=xaxis, c=c)


def _in_range(t: Trace, bounds: tuple[float, float], xaxis: str) -> Trace:
    return t.in_range(bounds=bounds, xaxis=xaxis)


@dataclass
class Config:
    # 0_0, 1_0, 2_0, 0_1, ..., 50_3
    id: int
    bracket: int | None
    params: dict[str, Any]

    def as_tuple(self) -> tuple[int, int | None]:
        return (self.id, self.bracket)

    def __str__(self) -> str:
        if self.bracket is None:
            return f"{self.id}"

        return f"{self.id}_{self.bracket}"

    def is_direct_continuation(self, of: Config) -> bool:
        if self.bracket is None:
            return False
        return self.id == of.id and of.bracket == self.bracket - 1

    @classmethod
    def parse(cls, dirname: str, config: dict) -> Config:
        config_name = dirname.replace("config_", "")
        if "_" in config_name:
            id_, bracket = map(int, config_name.split("_"))
        else:
            id_ = int(config_name)
            bracket = None

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

        info = result["info_dict"]
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
    results: list[Result]

    @classmethod
    def load(cls, path: Path, *, pool: Pool | None = None) -> Trace:
        trace_results_dir = path / "neps_root_directory" / "results"
        assert trace_results_dir.exists()
        config_dirs = [p for p in trace_results_dir.iterdir() if p.is_dir() and "config" in p.name]
        if pool:
            results = list(pool.imap_unordered(Result.from_dir, config_dirs))
        else:
            results = list(map(Result.from_dir, config_dirs))

        if len(results) == 0:
            raise ValueError(f"Couldn't find results in {trace_results_dir}")

        global_start = min(result.start_time for result in results)
        results = [
            result.mutate(
                start_time_since_global_start=result.start_time - global_start,
                end_time_since_global_start=result.end_time - global_start,
            )
            for result in results
        ]

        results = sorted(results, key=lambda r: r.end_time)
        return cls(results=results)

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
        df = df.set_index("end_time")
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
        def bracket(res: Result) -> int:
            return 0 if res.config.bracket is None else res.config.bracket

        results = {
            config_id: sorted(results, key=bracket)
            for config_id, results in groupby(self.results, key=lambda r: r.config.id)
        }

        continuations = []
        for config_results in results.values():
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
        xaxis: str,
        yaxis: str,
        *,
        op: Callable[[float, float], bool] = operator.lt,
    ) -> Trace:
        """Return a trace with only the incumbent results."""
        def _xaxis(r) -> float:
            return getattr(r, xaxis)

        def _yaxis(r) -> float:
            return getattr(r, yaxis)

        if op is not operator.lt and xaxis not in ("max_fidelity_loss", "loss"):
            raise NotImplementedError("Only supports lt with 'max_fidelity_loss' or 'loss'")

        results: list[Result] = sorted(self.results, key=_xaxis)
        incumbent = results[0]

        incumbents = [incumbent]
        for result in results[1:]:
            # If the new result is better than the incumbent, replace the incumbent
            if op(_yaxis(result), _yaxis(incumbent)):
                incumbent = result
                incumbents.append(incumbent)

        return replace(self, results=incumbents)

    def in_range(self, bounds: tuple[float, float], xaxis: str) -> Trace:
        low, high = bounds
        results = [
            result for result in self.results if low <= getattr(result, xaxis) <= high
        ]
        results = sorted(results, key=lambda r: getattr(r, xaxis))
        return replace(self, results=results)

    def rescale(self, c: float, xaxis: str) -> Trace:
        results: list[Result] = []
        for result in self.results:
            copied = replace(result)
            value = getattr(results, xaxis)
            setattr(copied, xaxis, value * c)
            results.append(copied)

        results = sorted(results, key=lambda r: getattr(r, xaxis))
        return replace(self, results=results)

    def series(self, index: str, values: str, name: str | None = None) -> pd.Series:
        indicies = [getattr(r, index) for r in self.results]
        vals = [getattr(r, values) for r in self.results]
        series = pd.Series(vals, index=indicies, name=name).sort_index()
        assert isinstance(series, pd.Series)
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
        expected_path = config_dir / f"{name}.yaml"
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
            prior_error=config["prior_highest_fidelity_error"],
            best_10_error=config["best_10_error"],
            best_25_error=config["best_25_error"],
            best_50_error=config["best_50_error"],
            best_90_error=config["best_90_error"],
            best_100_error=config["best_100_error"],
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


@dataclass
class AlgorithmResults(Mapping[int, Trace]):
    traces: dict[int, Trace]

    @classmethod
    def load(
        cls,
        path: Path,
        *,
        pool: Pool | None = None,
        seeds: list[int] | None = None,
    ) -> AlgorithmResults:
        """Load all traces for an algorithm."""
        if seeds is None:
            seeds = [
                int(p.name.split("=")[1])
                for p in path.iterdir()
                if p.is_dir() and "seed" in p.name
            ]

        traces = {seed: Trace.load(path / f"seed={seed}", pool=pool) for seed in seeds}
        return cls(traces=traces)

    def with_continuations(self, pool: Pool | None = None) -> AlgorithmResults:
        """Return a new AlgorithmResults with continuations."""
        traces: Iterable[Trace]
        if pool:
            traces = pool.imap(_with_continuations, self.traces.values())
        else:
            traces = map(_with_continuations, self.traces.values())

        itr = zip(self.traces.keys(), traces)
        return replace(self, traces={seed: trace for seed, trace in itr})

    def with_cumulative_fidelity(self, pool: Pool | None = None) -> AlgorithmResults:
        traces: Iterable[Trace]
        if pool:
            traces = pool.imap(_with_cumulative_fidelity, self.traces.values())
        else:
            traces = map(_with_cumulative_fidelity, self.traces.values())

        itr = zip(self.traces.keys(), traces)
        return replace(self, traces={seed: trace for seed, trace in itr})

    def incumbent_traces(
        self, xaxis: str, yaxis: str, pool: Pool | None = None
    ) -> AlgorithmResults:
        args = [(trace, xaxis, yaxis) for trace in self.traces.values()]
        traces: Iterable[Trace]
        if pool:
            traces = pool.starmap(_incumbent_trace, args)
        else:
            traces = starmap(_incumbent_trace, args)

        itr = zip(self.traces.keys(), traces)
        return replace(self, traces={seed: trace for seed, trace in itr})

    def rescale(
        self, xaxis: str, c: float, *, pool: Pool | None = None
    ) -> AlgorithmResults:
        args = [(trace, xaxis, c) for trace in self.traces.values()]
        traces: Iterable[Trace]
        if pool:
            traces = pool.starmap(_rescale, args)
        else:
            traces = starmap(_rescale, args)

        itr = zip(self.traces.keys(), traces)
        return replace(self, traces={seed: trace for seed, trace in itr})

    def in_range(
        self,
        bounds: tuple[float, float],
        xaxis: str,
        *,
        pool: Pool | None = None,
    ) -> AlgorithmResults:
        args = [(trace, bounds, xaxis) for trace in self.traces.values()]
        traces: Iterable[Trace]
        if pool:
            traces = pool.starmap(_in_range, args)
        else:
            traces = starmap(_in_range, args)

        itr = zip(self.traces.keys(), traces)
        return replace(self, traces={seed: trace for seed, trace in itr})

    def df(self, index: str, values: str) -> pd.DataFrame:
        """Return a dataframe with the traces."""
        columns = [
            trace.series(index=index, values=values, name=f"seed-{seed}")
            for seed, trace in self.traces.items()
        ]
        df = pd.concat(columns, axis=1).sort_index(ascending=True)

        assert isinstance(df, pd.DataFrame)
        return df

    def iter_results(self) -> Iterator[Result]:
        yield from chain.from_iterable(iter(trace) for trace in self.traces.values())

    def __getitem__(self, seed: int) -> Trace:
        return self.traces.__getitem__(seed)

    def __iter__(self) -> Iterator[int]:
        return self.traces.__iter__()

    def __len__(self) -> int:
        return self.traces.__len__()


@dataclass
class BenchmarkResults(Mapping[str, AlgorithmResults]):
    results: Mapping[str, AlgorithmResults]

    def with_continuations(self, pool: Pool | None = None) -> BenchmarkResults:
        results = {k: v.with_continuations(pool) for k, v in self.results.items()}
        return replace(self, results=results)

    def with_cumulative_fidelity(self, pool: Pool | None = None) -> BenchmarkResults:
        results = {k: v.with_cumulative_fidelity(pool) for k, v in self.results.items()}
        return replace(self, results=results)

    def incumbent_traces(self, xaxis: str, yaxis: str, *, pool: Pool | None = None) -> BenchmarkResults:
        results = {k: v.incumbent_traces(pool=pool, xaxis=xaxis, yaxis=yaxis) for k, v in self.results.items()}
        return replace(self, results=results)

    @classmethod
    def load(
        cls,
        path: Path,
        *,
        pool: Pool | None = None,
        algorithms: list[str] | None = None,
        seeds: list[int] | None = None,
    ) -> BenchmarkResults:
        if algorithms is None:
            algorithms = [
                p.name.split("=")[1]
                for p in path.iterdir()
                if p.is_dir() and "algo" in p.name
            ]

        results = {
            algo: AlgorithmResults.load(
                path / f"algorithm={algo}", seeds=seeds, pool=pool
            )
            for algo in algorithms
        }
        return cls(results)

    def rescale(
        self, xaxis: str, c: float, *, pool: Pool | None = None
    ) -> BenchmarkResults:
        results = {
            name: algo_results.rescale(xaxis=xaxis, c=c, pool=pool)
            for name, algo_results in self.results.items()
        }
        return replace(self, results=results)

    def in_range(
        self,
        bounds: tuple[float, float],
        xaxis: str,
        *,
        pool: Pool | None = None,
    ) -> BenchmarkResults:
        results = {
            name: algo_results.in_range(bounds=bounds, xaxis=xaxis, pool=pool)
            for name, algo_results in self.results.items()
        }
        return replace(self, results=results)

    def iter_results(self) -> Iterator[Result]:
        yield from chain.from_iterable(
            algo_results.iter_results() for algo_results in self.results.values()
        )

    def __getitem__(self, algo: str) -> AlgorithmResults:
        return self.results.__getitem__(algo)

    def __iter__(self) -> Iterator[str]:
        return self.results.__iter__()

    def __len__(self) -> int:
        return self.results.__len__()


@dataclass
class ExperimentResults(Mapping[str, BenchmarkResults]):
    benchmarks: dict[str, Benchmark]
    results: dict[str, BenchmarkResults]

    @classmethod
    def load(
        cls,
        path: Path,
        *,
        benchmarks: list[str] | None = None,
        algorithms: list[str] | None = None,
        seeds: list[int] | None = None,
        benchmark_config_dir: Path,
        pool: Pool | None = None,
    ) -> ExperimentResults:
        if benchmarks is None:
            benchmarks = [
                p.name.split("=")[1]
                for p in path.iterdir()
                if p.is_dir() and "benchmark" in p.name
            ]

        return cls(
            benchmarks={
                benchmark: Benchmark.from_name(benchmark, benchmark_config_dir)
                for benchmark in benchmarks
            },
            results={
                benchmark: BenchmarkResults.load(
                    path / f"benchmark={benchmark}",
                    algorithms=algorithms,
                    seeds=seeds,
                    pool=pool,
                )
                for benchmark in benchmarks
            },
        )

    def with_continuations(self, pool: Pool | None = None) -> ExperimentResults:
        results = {k: v.with_continuations(pool) for k, v in self.results.items()}
        return replace(self, results=results)

    def with_cumulative_fidelity(self, pool: Pool | None = None) -> ExperimentResults:
        results = {k: v.with_cumulative_fidelity(pool) for k, v in self.results.items()}
        return replace(self, results=results)

    def incumbent_trace(
        self,
        xaxis: str,
        yaxis: str,
        *,
        pool: Pool | None = None,
    ) -> ExperimentResults:
        results = {k: v.incumbent_traces(xaxis=xaxis, yaxis=yaxis, pool=pool) for k, v in self.results.items()}
        return replace(self, results=results)

    def rescale(
        self, xaxis: str, by: Literal["max_fidelity"], *, pool: Pool | None = None
    ) -> ExperimentResults:
        if by != "max_fidelity":
            raise NotImplementedError(f"by={by}")

        max_fidelities = {
            name: benchmark.max_fidelity for name, benchmark in self.benchmarks.items()
        }

        results = {
            name: benchmark_results.rescale(
                xaxis=xaxis, c=(1 / max_fidelities[name]), pool=pool
            )
            for name, benchmark_results in self.results.items()
        }
        return replace(self, results=results)

    def in_range(
        self, bounds: tuple[float, float], xaxis: str, *, pool: Pool | None = None
    ) -> ExperimentResults:
        results = {
            k: v.in_range(bounds=bounds, xaxis=xaxis, pool=pool)
            for k, v in self.results.items()
        }
        return replace(self, results=results)

    def __getitem__(self, benchmark: str) -> BenchmarkResults:
        return self.results.__getitem__(benchmark)

    def __iter__(self) -> Iterator[str]:
        return self.results.__iter__()

    def __len__(self) -> int:
        return self.results.__len__()

    def iter_results(self) -> Iterator[Result]:
        yield from chain.from_iterable(
            benchmark_results.iter_results()
            for benchmark_results in self.results.values()
        )
