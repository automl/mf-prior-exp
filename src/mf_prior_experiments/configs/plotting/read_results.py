import json
import os
from typing import List

import yaml  # type: ignore
from attrdict import AttrDict
from hpbandster.core.base_iteration import Datum
from hpbandster.core.result import Result

# default output format is assumed to be NePS
OUTPUT_FORMAT = {"hpbandster": ["bohb", "hpbandster"]}

SINGLE_FIDELITY_ALGORITHMS = [
    "random_search",
    "random_search_prior",
    "bayesian_optimization",
]


def load_yaml(filename):
    with open(filename, encoding="UTF-8") as f:
        # https://github.com/yaml/pyyaml/wiki/PyYAML-yaml.load(input)-Deprecation
        args = yaml.load(f, Loader=yaml.FullLoader)
    return AttrDict(args)


def logged_results_to_HBS_result(directory):
    """
    from hpbandster.core.result
    function to import logged 'live-results' and return a HB_result object


    You can load live run results with this function and the returned
    HB_result object gives you access to the results the same way
    a finished run would.

    Parameters
    ----------
    directory: str
        the directory containing the results.json and config.json files

    Returns
    -------
    hpbandster.core.result.Result: :object:
    """
    data = {}
    time_ref = float("inf")
    budget_set = set()

    with open(
        os.path.join(directory, "hpbandster_root_directory", "configs.json"),
        encoding="UTF-8",
    ) as f:
        for line in f:

            line = json.loads(line)

            if len(line) == 3:
                config_id, config, config_info = line
            if len(line) == 2:
                (
                    config_id,
                    config,
                ) = line
                config_info = "N/A"

            data[tuple(config_id)] = Datum(config=config, config_info=config_info)

    with open(
        os.path.join(directory, "hpbandster_root_directory", "results.json"),
        encoding="UTF-8",
    ) as f:
        for line in f:
            # try:
            config_id, budget, time_stamps, result, exception = json.loads(line)
            _id = tuple(config_id)

            data[_id].time_stamps[budget] = time_stamps
            data[_id].results[budget] = result
            data[_id].exceptions[budget] = exception

            budget_set.add(budget)
            time_ref = min(time_ref, time_stamps["submitted"])
            # except:
            #     continue

    # infer the hyperband configuration from the data
    budget_list = sorted(list(budget_set))

    HB_config = {
        "budgets": budget_list,
        "max_SH_iter": len(budget_set),
        "time_ref": time_ref,
    }
    return Result([data], HB_config)


def _get_info_neps(path, seed) -> List:
    with open(
        os.path.join(
            path, str(seed), "neps_root_directory", "all_losses_and_configs.txt"
        ),
        encoding="UTF-8",
    ) as f:
        data = f.readlines()
    losses = [
        float(entry.strip().split("Loss: ")[1]) for entry in data if "Loss: " in entry
    ]

    config_ids = [
        f"config_{entry.strip().split('Config ID: ')[1]}"
        for entry in data
        if "Config ID: " in entry
    ]
    info = []
    result_path = os.path.join(path, str(seed), "neps_root_directory", "results")
    for config_id in config_ids:
        result_yaml = load_yaml(os.path.join(result_path, config_id, "result.yaml"))
        start_time = (
            result_yaml.info_dict["start_time"]
            if "start_time" in result_yaml.info_dict
            else 0.0
        )
        end_time = (
            result_yaml.info_dict["end_time"]
            if "start_time" in result_yaml.info_dict
            else 0.0
        )
        info.append(
            dict(
                fidelity=result_yaml.info_dict["fidelity"],
                cost=result_yaml.info_dict["cost"],
                start_time=start_time,
                end_time=end_time,
                config_id=config_id,
            )
        )

    data = list(zip(config_ids, losses, info))  # type: ignore

    return data


def _get_info_hpbandster(path, seed) -> List:
    get_loss_from_run_fn = lambda r: r.loss
    # load runs from log file
    result = logged_results_to_HBS_result(os.path.join(path, str(seed)))
    # get all executed runs
    all_runs = result.get_all_runs()

    configs_evaluated = dict()  # type: ignore
    budgets = list(map(int, result.HB_config["budgets"]))

    data = []

    for run in all_runs:
        if run.loss is None:
            continue

        bracket_id, config_id = run.config_id[0], run.config_id[-1]
        budget_id = budgets.index(run.info["fidelity"])

        _previous_brackets_ids = 0
        if bracket_id > 0:
            for bracket in range(0, bracket_id):
                _previous_brackets_ids += len(configs_evaluated[bracket])
        config_id += _previous_brackets_ids

        if bracket_id in configs_evaluated:
            configs_evaluated[bracket_id].add(config_id)
        else:
            configs_evaluated[bracket_id] = {config_id}

        neps_config_id = f"config_{config_id}_{budget_id}"

        loss = get_loss_from_run_fn(run)

        data.append((neps_config_id, loss, run.info))

    return data


def _get_info_smac(path, seed):
    raise NotImplementedError("SMAC parsing not implemented!")


def get_seed_info(
    path, seed, cost_as_runtime=False, algorithm="random_search", n_workers=1
):
    """Reads and processes data per seed.

    An `algorithm` needs to be passed to calculate continuation costs.
    """

    if algorithm in OUTPUT_FORMAT["hpbandster"]:
        func = _get_info_hpbandster
    else:
        func = _get_info_neps
    data = func(path, seed)

    if n_workers == 1:
        key_to_extract = "cost" if cost_as_runtime else "fidelity"
        # max_cost only relevant for scaling x-axis when using fidelity on the x-axis
        max_cost = None if cost_as_runtime else 0
        if algorithm not in SINGLE_FIDELITY_ALGORITHMS:
            # calculates continuation costs for MF algorithms
            # NOTE: assumes that all recorded evaluations are black-box evaluations where
            #   continuations or freeze-thaw was not accounted for during optimization
            data.reverse()
            for idx, (data_id, loss, info) in enumerate(data):
                # `max_cost` tracks the maximum fidelity used for evaluation
                max_cost = (
                    max(max_cost, info[key_to_extract]) if max_cost is not None else None
                )
                for _id, _, _info in data[data.index((data_id, loss, info)) + 1 :]:
                    # if `_` is not found in the string, `split()` returns the original
                    # string and the 0-th element is the string itself, which fits the
                    # config ID format for non-NePS optimizers
                    # MF algos in NePS contain a 2-part ID separated by `_` with the first
                    # element denoting config ID and the second element denoting the rung
                    _subset_idx = 1 if "config" in data_id else 0
                    id_config_id = data_id.split("_")[_subset_idx]
                    _id_config_id = _id.split("_")[_subset_idx]
                    # checking if the base config ID is the same
                    if id_config_id != _id_config_id:
                        continue
                    # subtracting the immediate lower fidelity cost available from the
                    # current higher fidelity --> continuation cost
                    info[key_to_extract] -= _info[key_to_extract]
                    data[idx] = (data_id, loss, info)
                    break
            data.reverse()
        else:
            for idx, (data_id, loss, info) in enumerate(data):
                # `max_cost` tracks the maximum fidelity used for evaluation
                max_cost = (
                    max(max_cost, info[key_to_extract]) if max_cost is not None else None
                )
    else:
        global_start = data[0][-1]["start_time"]
        max_cost = None if cost_as_runtime else 0
        for idx, (data_id, loss, info) in enumerate(data):
            info["cost"] = info["end_time"] - global_start
            max_cost = max(max_cost, info["cost"]) if max_cost is not None else None
        max_cost += 10

    data = [(d[1], d[2]) for d in data]
    losses, infos = zip(*data)

    return list(losses), list(infos), max_cost
