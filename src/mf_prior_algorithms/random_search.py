from __future__ import annotations

import random

from metahyper.api import ConfigResult

from neps.optimizers.base_optimizer import BaseOptimizer
from neps.search_spaces import CategoricalParameter, FloatParameter, IntegerParameter
from neps.search_spaces.hyperparameters.categorical import CATEGORICAL_CONFIDENCE_SCORES
from neps.search_spaces.hyperparameters.constant import ConstantParameter
from neps.search_spaces.hyperparameters.float import FLOAT_CONFIDENCE_SCORES
from neps.search_spaces.search_space import SearchSpace

CUSTOM_FLOAT_CONFIDENCE_SCORES = FLOAT_CONFIDENCE_SCORES.copy()
CUSTOM_FLOAT_CONFIDENCE_SCORES.update({"ultra": 0.05})

CUSTOM_CATEGORICAL_CONFIDENCE_SCORES = CATEGORICAL_CONFIDENCE_SCORES.copy()
CUSTOM_CATEGORICAL_CONFIDENCE_SCORES.update({"ultra": 8})


class RandomSearch(BaseOptimizer):

    use_priors = False
    ignore_fidelity = True  # defaults to a black-box setup

    def __init__(self, random_interleave_prob: float = 0.0, **optimizer_kwargs):
        super().__init__(**optimizer_kwargs)
        self.random_interleave_prob = random_interleave_prob
        self._num_previous_configs: int = 0

    def load_results(
        self,
        previous_results: dict[str, ConfigResult],
        pending_evaluations: dict[str, ConfigResult],
    ) -> None:
        self._num_previous_configs = len(previous_results) + len(pending_evaluations)

    def get_config_and_ids(self) -> tuple[SearchSpace, str, str | None]:
        use_priors = (
            False if random.random() < self.random_interleave_prob else self.use_priors
        )
        config = self.pipeline_space.sample(
            patience=self.patience,
            user_priors=use_priors,
            ignore_fidelity=self.ignore_fidelity,
        )
        config_id = str(self._num_previous_configs + 1)
        return config.hp_values(), config_id, None


class RandomSearchWithPriors(RandomSearch):
    use_priors = True

    def __init__(self, prior_confidence: str = "medium", **optimizer_kwargs):
        super().__init__(**optimizer_kwargs)
        self.prior_confidence = prior_confidence
        self._enhance_priors()

    def _enhance_priors(self, confidence_score=None):
        """Only applicable when priors are given along with a confidence.

        Args:
            confidence_score: dict
                The confidence scores for the types.
                Example: {"categorical": 5.2, "numeric": 0.15}
        """
        if not self.use_priors and self.prior_confidence is None:
            return
        for k, v in self.pipeline_space.items():
            if v.is_fidelity or isinstance(v, ConstantParameter):
                continue
            elif isinstance(v, (FloatParameter, IntegerParameter)):
                if confidence_score is None:
                    confidence = CUSTOM_FLOAT_CONFIDENCE_SCORES[self.prior_confidence]
                else:
                    confidence = confidence_score["numeric"]
                self.pipeline_space[k].default_confidence_score = confidence
            elif isinstance(v, CategoricalParameter):
                if confidence_score is None:
                    confidence = CUSTOM_CATEGORICAL_CONFIDENCE_SCORES[
                        self.prior_confidence
                    ]
                else:
                    confidence = confidence_score["categorical"]
                self.pipeline_space[k].default_confidence_score = confidence
