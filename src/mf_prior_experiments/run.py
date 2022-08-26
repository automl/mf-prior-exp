import contextlib
import logging
import random
import sys
from pathlib import Path

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
    import neps

    benchmark = hydra.utils.instantiate(args.benchmark.api)

    def run_pipeline(**config):
        config = benchmark.sample()  # TODO use the config provided by neps
        return benchmark.query(config).valid_acc

    pipeline_space = dict(
        search_space=benchmark.space,
        epoch=neps.IntegerParameter(lower=1, upper=200, is_fidelity=True),
    )
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

    print(args.benchmark.api.datadir)

    hydra.utils.call(args.algorithm.run_function, args)
    logger.info("Run finished")


if __name__ == "__main__":
    run()  # pylint: disable=no-value-for-parameter
