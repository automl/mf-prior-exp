from __future__ import annotations

from metahyper.api import ConfigResult

from neps.optimizers.base_optimizer import BaseOptimizer
from neps.search_spaces import CategoricalParameter, FloatParameter, IntegerParameter
from neps.search_spaces.search_space import SearchSpace


class RandomSearch(BaseOptimizer):

    use_priors = False
    ignore_fidelity = True  # defaults to a black-box setup

    def __init__(self, **optimizer_kwargs):
        super().__init__(**optimizer_kwargs)
        self._num_previous_configs: int = 0

    def load_results(
        self,
        previous_results: dict[str, ConfigResult],
        pending_evaluations: dict[str, ConfigResult],
    ) -> None:
        self._num_previous_configs = len(previous_results) + len(pending_evaluations)

    def get_config_and_ids(self) -> tuple[SearchSpace, str, str | None]:
        config = self.pipeline_space.sample(
            patience=self.patience,
            user_priors=self.use_priors,
            ignore_fidelity=self.ignore_fidelity,
        )
        config_id = str(self._num_previous_configs + 1)
        return config.hp_values(), config_id, None


class RandomSearchWithPriors(RandomSearch):
    use_priors = True

    def __init__(self, **optimizer_kwargs):
        super().__init__(**optimizer_kwargs)
        # for Prior based optimizers, assume confident priors
        self.confidence_scores = {
            "categorical": 4,  # 2.5,
            "numeric": 0.05,  # 0.125,
        }
        self._enhance_priors()

    def _enhance_priors(self):
        for k in self.pipeline_space.keys():
            if self.pipeline_space[k].is_fidelity:
                continue
            if isinstance(self.pipeline_space[k], (FloatParameter, IntegerParameter)):
                confidence = self.confidence_scores["numeric"]
                self.pipeline_space[k].default_confidence_score = confidence
            elif isinstance(self.pipeline_space[k], CategoricalParameter):
                confidence = self.confidence_scores["categorical"]
                self.pipeline_space[k].default_confidence_score = confidence
