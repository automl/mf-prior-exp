import contextlib
import logging
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

import hydra
import numpy as np
import yaml
from gitinfo import gitinfo
from omegaconf import OmegaConf

logger = logging.getLogger("mf_prior_experiments.run")

# NOTE: If editing this, please look for MIN_SLEEP_TIME
# in `read_results.py` and change it there too
MIN_SLEEP_TIME = 10  # 10s hopefully is enough to simulate wait times for metahyper

# Use this environment variable to force overwrite when running
OVERWRITE = False  # bool(os.environ.get("MF_EXP_OVERWRITE", False))

print(f"{'='*50}\noverwrite={OVERWRITE}\n{'='*50}")


def _set_seeds(seed):
    random.seed(seed)  # important for NePS optimizers
    np.random.seed(seed)  # important for NePS optimizers
    # torch.manual_seed(seed)
    # torch.backends.cudnn.benchmark = False
    # torch.backends.cudnn.deterministic = True
    # torch.manual_seed(seed)
    # tf.random.set_seed(seed)


def run_dyhpo(args):
    import ConfigSpace as cs
    from mfpbench import Benchmark
    from mfpbench.result import Result

    # messy import of DyHPO
    from DyHPO.hpo_method import DyHPOAlgorithm

#   # TODO
#   class Dyhpo:
#       def __init__(self, *args, **kwargs):
#           pass

    benchmark: Benchmark = hydra.utils.instantiate(args.benchmark.api)  # type: ignore

    # TODO
    # assert all hps are purely numerical
    # Convert all to purely numerical array
    space = benchmark.space
    # grid_of_configuration_arrays = to_grid(space)
    grid_of_configurations = space.sample_configuration(8000)
    grid_of_configuration_arrays = [c.get_array() for c in grid_of_configurations]

    TOTAL_BUDGET = 10 # TODO

    dhypo = DyHPOAlgorithm(
        hp_candidates=np.asarray(grid_of_configuration_arrays),
        log_indicator=[hp.log for hp in space.get_hyperparameters()],
        seed=args.seed,
        max_benchmark_epochs=benchmark.end,
        fantasize_step=args.algorithm.dyhpo_args.fantasize_step,
        minimization=args.algorithm.dyhpo_args.minimization,  # TODO: verify
        total_budget=TOTAL_BUDGET,
        device=args.algorithm.dyhpo_args.device, # Maybe GPU?
        dataset_name=args.benchmark.name,
        output_path="TODO",
        surrogate_config=args.algorithm.dyhpo_args.surrogate_config,
        verbose=args.algorithm.dyhpo_args.verbose,
    )

    # NOTE (eddiebergman,heri): Seems Dhypo's benchmarks
    #    actually start at epoch 1 instead of 0. This means
    #    that they have signal initially. A bit cheaty.
    for _ in range(TOTAL_BUDGET):

        start = time.time()

        hp_index, budget = dhypo.suggest()

        config = grid_of_configurations[hp_index]

        results: list[Result] = benchmark.trajectory(config, to=budget)
        max_fidelity_result = benchmark.query(config, at=benchmark.end)

        result = results[-1]

        if len(results) > 1:
            previous_result = results[-2]
            continuation_fidelity = result.fidelity - previous_result.fidelity
        else:
            continuation_fidelity = None

        learning_curve = [r.error for r in results]

        dhypo.observe(hp_index=hp_index, b=budget, learning_curve=learning_curve)

        end = time.time()

        result_for_disk = {
            "loss": result.error,
            "cost": result.cost,
            "info_dict": {
                "cost": result.cost,
                "fidelity": result.fidelity,
                "val_score": result.val_score,
                "test_score": result.test_score,
                "continuation_fidelity": continuation_fidelity,
                "start_time": start,
                "end_time": end,  # + fidelity,
                "max_fidelity_loss": float(max_fidelity_result.error),
                "max_fidelity_cost": float(max_fidelity_result.cost),
                "process_id": os.getpid(),
                # val_error: result.val_error
                # test_error: result.test_error
            },
        }

        # TODO: Save to disk






def run_hpbandster(args):
    import uuid

    import ConfigSpace
    import hpbandster.core.nameserver as hpns
    import hpbandster.core.result as hpres
    from hpbandster.core.worker import Worker
    from hpbandster.optimizers.bohb import BOHB
    from hpbandster.optimizers.hyperband import HyperBand
    from mfpbench import Benchmark

    # Added the type here just for editors to be able to get a quick view
    benchmark: Benchmark = hydra.utils.instantiate(args.benchmark.api)

    def compute(**config: Any) -> dict:
        fidelity = config["budget"]
        config = config["config"]

        # transform to Ordinal HPs back
        for hp_name, hp in benchmark.space.items():
            if isinstance(hp, ConfigSpace.OrdinalHyperparameter):
                config[hp_name] = hp.sequence[config[hp_name] - 1]

        result = benchmark.query(config, at=int(fidelity))

        # This design only makes sense in the context of surrogate/tabular
        # benchmarks, where we do not actually need to run the model being
        # queried.
        max_fidelity_result = benchmark.query(config, at=benchmark.end)

        # we need to cast to float here as serpent will break on np.floating that might
        # come from a benchmark (LCBench)
        return {
            "loss": float(result.error),
            "cost": float(result.cost),
            "info": {
                "cost": float(result.cost),
                "val_score": float(result.val_score),
                "test_score": float(result.test_score),
                "fidelity": float(result.fidelity)
                if isinstance(result.fidelity, np.floating)
                else result.fidelity,
                "max_fidelity_loss": float(max_fidelity_result.error),
                "max_fidelity_cost": float(max_fidelity_result.cost),
                "process_id": os.getpid(),
                # val_error: result.val_error
                # test_error: result.test_error
            },
        }

    lower, upper, _ = benchmark.fidelity_range
    fidelity_name = benchmark.fidelity_name
    benchmark_configspace = benchmark.space

    # BOHB does not accept Ordinal HPs
    bohb_configspace = ConfigSpace.ConfigurationSpace(
        name=benchmark_configspace.name, seed=args.seed
    )

    for hp_name, hp in benchmark_configspace.items():
        if isinstance(hp, ConfigSpace.OrdinalHyperparameter):
            int_hp = ConfigSpace.UniformIntegerHyperparameter(
                hp_name, lower=1, upper=len(hp.sequence)
            )
            bohb_configspace.add_hyperparameters([int_hp])
        else:
            bohb_configspace.add_hyperparameters([hp])

    logger.info(f"Using configspace: \n {benchmark_configspace}")
    logger.info(f"Using fidelity: \n {fidelity_name} in {lower}-{upper}")

    max_evaluations_total = 10

    run_id = str(uuid.uuid4())
    NS = hpns.NameServer(
        run_id=run_id, port=0, working_directory="hpbandster_root_directory"
    )
    ns_host, ns_port = NS.start()

    hpbandster_worker = Worker(nameserver=ns_host, nameserver_port=ns_port, run_id=run_id)
    hpbandster_worker.compute = compute
    hpbandster_worker.run(background=True)

    result_logger = hpres.json_result_logger(
        directory="hpbandster_root_directory", overwrite=True
    )
    hpbandster_config = {
        "eta": 3,
        "min_budget": lower,
        "max_budget": upper,
        "run_id": run_id,
    }

    if "model" in args.algorithm and args.algorithm.model:
        hpbandster_cls = BOHB
    else:
        hpbandster_cls = HyperBand

    hpbandster_optimizer = hpbandster_cls(
        configspace=bohb_configspace,
        nameserver=ns_host,
        nameserver_port=ns_port,
        result_logger=result_logger,
        **hpbandster_config,
    )

    logger.info("Starting run...")
    res = hpbandster_optimizer.run(n_iterations=max_evaluations_total)

    hpbandster_optimizer.shutdown(shutdown_workers=True)
    NS.shutdown()

    id2config = res.get_id2config_mapping()
    logger.info(f"A total of {len(id2config.keys())} queries.")


def run_neps(args):
    from mfpbench import Benchmark

    import neps

    # Added the type here just for editors to be able to get a quick view
    benchmark: Benchmark = hydra.utils.instantiate(args.benchmark.api)  # type: ignore

    def run_pipeline(previous_pipeline_directory: Path, **config: Any) -> dict:
        start = time.time()
        if benchmark.fidelity_name in config:
            fidelity = config.pop(benchmark.fidelity_name)
        else:
            fidelity = benchmark.fidelity_range[1]

        result = benchmark.query(config, at=fidelity)

        # This design only makes sense in the context of surrogate/tabular
        # benchmarks, where we do not actually need to run the model being
        # queried.
        max_fidelity_result = benchmark.query(config, at=benchmark.end)

        # To account for continuations of previous configs in the parallel setting,
        # we use the `previous_pipeline_directory` which indicates if there has been
        # a previous lower fidelity evaluation of this config. If that's the case we
        # then subtract the previous fidelity off of this current one to compute
        # the `continuation_fidelity`. Otherwise, the `continuation_fidelity` is
        # just the current one. This is then used to make the worker sleep and
        # so we get a hueristically correct setup for each worker. In contrast,
        # if we do not do this, workers will not have even close to the correct
        # timestamps, and the order in which workers pick up new configurations to
        # evaluate may be in a very different order than if done in a real context.
        if args.n_workers == 1:
            # In the single worker setting, this does not matter and we can use
            # post-processing of the results to calculate the `continuation_fidelity`.
            continuation_fidelity = None
        else:
            # If there's no previous config, we sleep for `fidelity`.
            if previous_pipeline_directory is None:
                continuation_fidelity = None
                fidelity_sleep_time = fidelity

            # If there is a previous config, we calculate the `continuation_fidelity`
            # and sleep for that time instead
            else:
                previous_results_file = previous_pipeline_directory / "result.yaml"
                with previous_results_file.open("r") as f:
                    previous_results = yaml.load(f, Loader=yaml.FullLoader)

                # Calculate the continuation fidelity for sleeping
                current_fidelity = fidelity
                previous_fidelity = previous_results["info_dict"]["fidelity"]
                continuation_fidelity = current_fidelity - previous_fidelity

                logger.info("-"*30)
                logger.info(f"Continuing from: {previous_pipeline_directory}")
                logger.info(f"`continuation_fidelity`={continuation_fidelity}`")
                logger.info(f"{previous_results}")
                logger.info("-"*30)


                fidelity_sleep_time = continuation_fidelity

            time.sleep(fidelity_sleep_time + MIN_SLEEP_TIME)

        end = time.time()
        return {
            "loss": result.error,
            "cost": result.cost,
            "info_dict": {
                "cost": result.cost,
                "val_score": result.val_score,
                "test_score": result.test_score,
                "fidelity": result.fidelity,
                "continuation_fidelity": continuation_fidelity,
                "start_time": start,
                "end_time": end,  # + fidelity,
                "max_fidelity_loss": float(max_fidelity_result.error),
                "max_fidelity_cost": float(max_fidelity_result.cost),
                "process_id": os.getpid(),
                # val_error: result.val_error
                # test_error: result.test_error
            },
        }

    lower, upper, _ = benchmark.fidelity_range
    fidelity_name = benchmark.fidelity_name

    pipeline_space = {"search_space": benchmark.space}
    if args.algorithm.mf:
        if isinstance(lower, float):
            fidelity_param = neps.FloatParameter(
                lower=lower, upper=upper, is_fidelity=True
            )
        else:
            fidelity_param = neps.IntegerParameter(
                lower=lower, upper=upper, is_fidelity=True
            )
        pipeline_space = {**pipeline_space, **{fidelity_name: fidelity_param}}
        logger.info(f"Using fidelity space: \n {fidelity_param}")
    # pipeline_space = {"search_space": benchmark.space, fidelity_name: fidelity_param}
    logger.info(f"Using search space: \n {pipeline_space}")

    # TODO: could we pass budget per benchmark
    # if "budget" in args.benchmark:
    #     budget_args = {"budget": args.benchmark.budget}
    # else:
    #     budget_args = {"max_evaluations_total": 50}

    if "mf" in args.algorithm and args.algorithm.mf:
        max_evaluations_total = 130
    else:
        max_evaluations_total = 25

    neps.run(
        run_pipeline=run_pipeline,
        pipeline_space=pipeline_space,
        root_directory="neps_root_directory",
        # TODO: figure out how to pass runtime budget and if metahyper internally
        #  calculates continuation costs to subtract from optimization budget
        # **budget_args,
        max_evaluations_total=max_evaluations_total,
        searcher=hydra.utils.instantiate(args.algorithm.searcher, _partial_=True),
        overwrite_working_directory=OVERWRITE,
    )


@hydra.main(config_path="configs", config_name="run", version_base="1.2")
def run(args):
    # Print arguments to stderr (useful on cluster)
    print("run()")
    sys.stderr.write(f"{' '.join(sys.argv)}\n")
    sys.stderr.write(f"args = {args}\n\n")
    sys.stderr.flush()

    _set_seeds(args.seed)
    working_directory = Path().cwd()

    # Log general information
    logger.info(f"Using working_directory={working_directory}")
    with contextlib.suppress(TypeError):
        git_info = gitinfo.get_git_info()
        logger.info(f"Commit hash: {git_info['commit']}")
        logger.info(f"Commit date: {git_info['author_date']}")
    logger.info(f"Arguments:\n{OmegaConf.to_yaml(args)}")

    # Actually run
    hydra.utils.call(args.algorithm.run_function, args)
    logger.info("Run finished")


if __name__ == "__main__":
    run()  # pylint: disable=no-value-for-parameter
