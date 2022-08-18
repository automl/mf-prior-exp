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
    # Using hydra's utilities such as hydra.utils.call
    logger.info("Run finished")


if __name__ == "__main__":
    run()  # pylint: disable=no-value-for-parameter
