import itertools
from numbers import Real
from typing import Any, Dict, List, Mapping, Optional, Sequence, Union
import warnings
import optuna
import numpy as np
from scipy.stats import qmc

from optuna.distributions import BaseDistribution
from optuna.logging import get_logger
from optuna.samplers import BaseSampler
from optuna.study import Study
from optuna.trial import FrozenTrial, TrialState
from optuna.samplers._lazy_random_state import LazyRandomState

GridValueType = Union[str, float, int, bool, None]

_logger = get_logger(__name__)

class UniformDesignSampler(BaseSampler):
    def __init__(
        self, search_space: Mapping[str, Sequence[GridValueType]], seed: Optional[int] = None
    ) -> None:
        for param_name, param_values in search_space.items():
            for value in param_values:
                self._check_value(param_name, value)

        self._search_space = {}
        for param_name, param_values in sorted(search_space.items()):
            self._search_space[param_name] = list(param_values)

        self._param_names = sorted(search_space.keys())
        self._num_params = len(self._param_names)
        self._num_levels = [len(values) for values in self._search_space.values()]
        self._n_min_trials = np.prod(self._num_levels)


#ここから修正箇所
        self._rng = LazyRandomState(seed) #指定されたseed値のLazyRandomStateで乱数生成器を初期化
        self._qmc_rng = qmc.LatinHypercube(d=self._num_params, seed=self._rng.rng) #その値をLHCのseedに指定
        self._samples = self._qmc_rng.random(n=self._n_min_trials)

        self._current_trial = 0

    def reseed_rng(self) -> None:
        self._qmc_rng = qmc.LatinHypercube(d=self._num_params, seed=self._rng.rng) #その値をLHCのseedに指定
        self._samples = self._qmc_rng.random(n=self._n_min_trials)
#ここまで修正箇所



    def before_trial(self, study: Study, trial: FrozenTrial) -> None:
        if "grid_id" in trial.system_attrs or "fixed_params" in trial.system_attrs:
            return

        if self._current_trial < self._n_min_trials:
            study._storage.set_trial_system_attr(
                trial._trial_id, "search_space", self._search_space
            )
            study._storage.set_trial_system_attr(trial._trial_id, "grid_id", self._current_trial)
            self._current_trial += 1
            return

        target_grids = self._get_unvisited_grid_ids(study)

        if len(target_grids) == 0:
            _logger.warning(
                "`UniformDesignSampler` is re-evaluating a configuration because the grid has been "
                "exhausted. This may happen due to a timing issue during distributed optimization "
                "or when re-running optimizations on already finished studies."
            )
            target_grids = list(range(self._n_min_trials))

        grid_id = int(self._rng.rng.choice(target_grids))

        study._storage.set_trial_system_attr(trial._trial_id, "search_space", self._search_space)
        study._storage.set_trial_system_attr(trial._trial_id, "grid_id", grid_id)

    def infer_relative_search_space(
        self, study: Study, trial: FrozenTrial
    ) -> Dict[str, BaseDistribution]:
        return {}

    def sample_relative(
        self, study: Study, trial: FrozenTrial, search_space: Dict[str, BaseDistribution]
    ) -> Dict[str, Any]:
        return {}

    def sample_independent(
        self,
        study: Study,
        trial: FrozenTrial,
        param_name: str,
        param_distribution: BaseDistribution,
    ) -> Any:
        if "grid_id" not in trial.system_attrs:
            message = "All parameters must be specified when using UniformDesignSampler with enqueue_trial."
            raise ValueError(message)

        if param_name not in self._search_space:
            message = "The parameter name, {}, is not found in the given grid.".format(param_name)
            raise ValueError(message)

        grid_id = trial.system_attrs["grid_id"]
        sample_idx = self._param_names.index(param_name)
        sample_value_idx = int(self._samples[grid_id][sample_idx] * self._num_levels[sample_idx])
        param_value = self._search_space[param_name][sample_value_idx]

        contains = param_distribution._contains(param_distribution.to_internal_repr(param_value))
        if not contains:
            warnings.warn(
                f"The value `{param_value}` is out of range of the parameter `{param_name}`. "
                f"The value will be used but the actual distribution is: `{param_distribution}`."
            )

        return param_value

    def after_trial(
        self,
        study: Study,
        trial: FrozenTrial,
        state: TrialState,
        values: Optional[Sequence[float]],
    ) -> None:
        target_grids = self._get_unvisited_grid_ids(study)

        if len(target_grids) == 0:
            study.stop()
        elif len(target_grids) == 1:
            grid_id = study._storage.get_trial_system_attrs(trial._trial_id)["grid_id"]
            if grid_id == target_grids[0]:
                study.stop()

    @staticmethod
    def _check_value(param_name: str, param_value: Any) -> None:
        if param_value is None or isinstance(param_value, (str, int, float, bool)):
            return

        message = (
            "{} contains a value with the type of {}, which is not supported by "
            "`UniformDesignSampler`. Please make sure a value is `str`, `int`, `float`, `bool`"
            " or `None` for persistent storage.".format(param_name, type(param_value))
        )
        warnings.warn(message)

    def _get_unvisited_grid_ids(self, study: Study) -> List[int]:
        visited_grids = []
        running_grids = []

        trials = study._storage.get_all_trials(study._study_id, deepcopy=False)

        for t in trials:
            if "grid_id" in t.system_attrs and self._same_search_space(
                t.system_attrs["search_space"]
            ):
                if t.state.is_finished():
                    visited_grids.append(t.system_attrs["grid_id"])
                elif t.state == TrialState.RUNNING:
                    running_grids.append(t.system_attrs["grid_id"])

        unvisited_grids = set(range(self._n_min_trials)) - set(visited_grids) - set(running_grids)

        if len(unvisited_grids) == 0:
            unvisited_grids = set(range(self._n_min_trials)) - set(visited_grids)

        return list(unvisited_grids)

    @staticmethod
    def _grid_value_equal(value1: GridValueType, value2: GridValueType) -> bool:
        value1_is_nan = isinstance(value1, Real) and np.isnan(float(value1))
        value2_is_nan = isinstance(value2, Real) and np.isnan(float(value2))
        return (value1 == value2) or (value1_is_nan and value2_is_nan)

    def _same_search_space(self, search_space: Mapping[str, Sequence[GridValueType]]) -> bool:
        if set(search_space.keys()) != set(self._search_space.keys()):
            return False

        for param_name in search_space.keys():
            if len(search_space[param_name]) != len(self._search_space[param_name]):
                return False

            for i, param_value in enumerate(search_space[param_name]):
                if not self._grid_value_equal(param_value, self._search_space[param_name][i]):
                    return False

        return True