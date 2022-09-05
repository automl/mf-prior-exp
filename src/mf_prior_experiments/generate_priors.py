import contextlib
import logging
import os
import random
import sys
from pathlib import Path

import hydra
import numpy as np
from gitinfo import gitinfo
from omegaconf import OmegaConf

from neps.search_spaces.search_space import SearchSpace

logger = logging.getLogger("mf_prior_experiments.run")


def _set_seeds(seed):
    random.seed(seed)
    # np.random.seed(seed)
    # torch.manual_seed(seed)
    # torch.backends.cudnn.benchmark = False
    # torch.backends.cudnn.deterministic = True
    # torch.manual_seed(seed)
    # tf.random.set_seed(seed)


def run_for_priors(args):
    from mfpbench import Benchmark

    import neps

    # Added the type here just for editors to be able to get a quick view
    benchmark: Benchmark = hydra.utils.instantiate(args.benchmark.api)

    configs = []
    scores = []
    for i in range(args.nsamples):
        print(f"{i+1}/{args.nsamples}")
        config = benchmark.space.sample_configuration().get_dictionary()
        fidelity = benchmark.fidelity_range[1]
        result = benchmark.query(config, at=fidelity)
        configs.append(config)
        scores.append(result.error)

    good_idx = np.argmin(scores)
    bad_idx = np.argmax(scores)

    save_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "../../", "results_prior"
    )
    os.makedirs(save_dir, exist_ok=True)
    print("\n\nGood Prior:")
    print(configs[good_idx], scores[good_idx])
    with open(
        os.path.join(save_dir, f"{args.benchmark.name}_good.yaml"), "w", encoding="utf-8"
    ) as f:
        OmegaConf.save(config=OmegaConf.create(configs[good_idx]), f=f)
    print("\nBad Prior:")
    print(configs[bad_idx], scores[bad_idx])
    with open(
        os.path.join(save_dir, f"{args.benchmark.name}_bad.yaml"), "w", encoding="utf-8"
    ) as f:
        OmegaConf.save(config=OmegaConf.create(configs[good_idx]), f=f)
    return


@hydra.main(config_path="configs", config_name="run_for_priors", version_base="1.2")
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
    hydra.utils.call(args.run_function, args)
    logger.info("Run finished")


if __name__ == "__main__":
    run()  # pylint: disable=no-value-for-parameter
