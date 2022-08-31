import contextlib
import logging
import random
import sys
from pathlib import Path
from typing import Any

import hydra
from gitinfo import gitinfo
from omegaconf import OmegaConf

logger = logging.getLogger("mf_prior_experiments.run")


def _set_seeds(seed):
    random.seed(seed)
    # np.random.seed(seed)
    # torch.manual_seed(seed)
    # torch.backends.cudnn.benchmark = False
    # torch.backends.cudnn.deterministic = True
    # torch.manual_seed(seed)
    # tf.random.set_seed(seed)


def run_neps(args):
    from mfpbench import Benchmark

    import neps

    # Added the type here just for editors to be able to get a quick view
    benchmark: Benchmark = hydra.utils.instantiate(args.benchmark.api)

    def run_pipeline(**config: Any) -> dict:
        fidelity = config.pop(benchmark.fidelity_name)
        result = benchmark.query(config, at=fidelity)
        return {
            "loss": result.error,
            "info_dict": {
                "cost": result.train_time,
                "val_score": result.val_score,
                "test_score": result.test_score,
                "fidelity": result.fidelity,
            },
        }

    lower, upper, _ = benchmark.fidelity_range
    fidelity_name = benchmark.fidelity_name

    if isinstance(lower, float):
        fidelity_param = neps.FloatParameter(lower=lower, upper=upper, is_fidelity=True)
    else:
        fidelity_param = neps.IntegerParameter(lower=lower, upper=upper, is_fidelity=True)

    pipeline_space = {"search_space": benchmark.space, fidelity_name: fidelity_param}
    logger.info(f"Using search space: \n {pipeline_space}")

    neps.run(
        run_pipeline=run_pipeline,
        pipeline_space=pipeline_space,
        root_directory="neps_root_directory",
        max_evaluations_total=10,  # TODO use budget defined by benchmark
        searcher=hydra.utils.instantiate(args.algorithm.searcher, _partial_=True),
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
