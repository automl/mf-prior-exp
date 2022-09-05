from typing import Any, Union

from typing_extensions import Literal

from neps.optimizers.multi_fidelity.promotion_policy import AsyncPromotionPolicy
from neps.optimizers.multi_fidelity.sampling_policy import FixedPriorPolicy
from neps.optimizers.multi_fidelity.successive_halving import (
    AsynchronousSuccessiveHalvingWithPriors,
)
from neps.search_spaces.search_space import SearchSpace


class PriorWeightedPromotionPolicy(AsyncPromotionPolicy):
    """Extends asynchronous promotion to be weighted by the prior.

    Ranking is based on the prior-weighted score of p(x) times y(x).
    """

    def __init__(self, eta, **kwargs):
        super().__init__(eta, **kwargs)
        self.prior_model = None
        self.y_star = None

    def set_state(
        self,
        *,  # allows only keyword args
        max_rung: int,
        members: dict,
        performances: dict,
        config_map: dict,
        prior_model: Any,
        y_star: Union[float, int],
        **kwargs,  # pylint: disable=unused-argument
    ) -> None:
        super().set_state(
            max_rung=max_rung,
            members=members,
            performances=performances,
            config_map=config_map,
        )
        self.prior_model = prior_model
        self.y_star = y_star


class OurOptimizerV1(AsynchronousSuccessiveHalvingWithPriors):
    """Implements a SuccessiveHalving procedure with a sampling and promotion policy."""

    def __init__(
        self,
        pipeline_space: SearchSpace,
        budget: int,
        eta: int = 3,
        early_stopping_rate: int = 0,
        initial_design_type: Literal["max_budget", "unique_configs"] = "max_budget",
        sampling_policy: Any = FixedPriorPolicy,
        promotion_policy: Any = AsyncPromotionPolicy,
        loss_value_on_error: float = None,
        cost_value_on_error: float = None,
        logger=None,
    ):
        super().__init__(
            pipeline_space=pipeline_space,
            budget=budget,
            eta=eta,
            early_stopping_rate=early_stopping_rate,
            initial_design_type=initial_design_type,
            sampling_policy=sampling_policy,
            promotion_policy=promotion_policy,
            loss_value_on_error=loss_value_on_error,
            cost_value_on_error=cost_value_on_error,
            logger=logger,
        )
