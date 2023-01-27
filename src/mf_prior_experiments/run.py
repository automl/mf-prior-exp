import contextlib
import logging
import random
import os
import sys
import time
from pathlib import Path
from typing import Any

import hydra
import numpy as np
from gitinfo import gitinfo
from omegaconf import OmegaConf

logger = logging.getLogger("mf_prior_experiments.run")

# NOTE: If editing this, please look for MIN_SLEEP_TIME
# in `read_results.py` and change it there too
MIN_SLEEP_TIME = 10  # 10s hopefully is enough to simulate wait times for metahyper

# Use this environment variable to force overwrite when running
OVERWRITE = False # bool(os.environ.get("MF_EXP_OVERWRITE", False))

print(f"{'='*50}\noverwrite={OVERWRITE}\n{'='*50}")


def _set_seeds(seed):
    random.seed(seed)  # important for NePS optimizers
    np.random.seed(seed)  # important for NePS optimizers
    # torch.manual_seed(seed)
    # torch.backends.cudnn.benchmark = False
    # torch.backends.cudnn.deterministic = True
    # torch.manual_seed(seed)
    # tf.random.set_seed(seed)


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
    benchmark: Benchmark = hydra.utils.instantiate(args.benchmark.api)

    def run_pipeline(**config: Any) -> dict:
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

        if args.n_workers > 1:
            # essential step to simulate speed-up
            time.sleep(fidelity + MIN_SLEEP_TIME)

        end = time.time()
        return {
            "loss": result.error,
            "cost": result.cost,
            "info_dict": {
                "cost": result.cost,
                "val_score": result.val_score,
                "test_score": result.test_score,
                "fidelity": result.fidelity,
                "start_time": start,
                "end_time": end,  # + fidelity,
                "max_fidelity_loss": float(max_fidelity_result.error),
                "max_fidelity_cost": float(max_fidelity_result.cost),
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
        max_evaluations_total = 125
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
