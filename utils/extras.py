from copy import deepcopy
from pathlib import Path
from typing import Union

import numpy as np
import yaml
from ray import tune
from scipy.optimize import linear_sum_assignment

EPS = 1e-15
MAX_LOGSTD = 10

TARGET_METRICS = {
    "link_prediction": "LINK_PREDICTION_auc",
    "classification": "CLASSIFICATION_f1_mic",
    "clustering": "CLUSTERING_nmi"
}


def map_labels(Y_pred, Y):
    assert Y_pred.size == Y.size
    D = max(Y_pred.max(), Y.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(Y_pred.size):
        w[Y_pred[i], Y[i]] += 1
    ind = linear_sum_assignment(w.max() - w)
    return ind


def dict_to_print_str(kvs):
    return "".join(["{}: {:.4f}\t".format(str(k), v) for k, v in kvs.items()])


def two_level_dict_to_print_str(kvs):
    return "".join("||" + k.upper() + " => " + v + "\t" for k, v in
                   {k: dict_to_print_str(kvs[k]) for k, v in kvs.items() if v != {}}.items())


def join_two_level_dict_to_one_level_dict(kvs):
    return {k1.upper() + "_" + k2: kvs[k1][k2] for k1 in kvs.keys() for k2 in kvs[k1].keys()}


def load_config(config_file_path: Path = None):
    def deep_update(base_dict: dict, delta_dict: dict):
        return_dict = deepcopy(base_dict)
        for k in delta_dict:
            if isinstance(delta_dict[k], dict):
                return_dict[k] = deep_update(return_dict[k], delta_dict[k])
            else:
                return_dict[k] = delta_dict[k]
        return return_dict

    if config_file_path is None:
        config_file_path = Path(__file__).parent.parent / "dataset_config.yaml"
    config_all = yaml.safe_load(config_file_path.open("r"))

    all_datasets = config_all.get("target_datasets")
    if all_datasets is None:
        all_datasets = config_all["available_datasets"]

    config_base = deepcopy(config_all)["base"]
    return_config = deepcopy(config_all)

    for target_dataset in all_datasets:
        return_config[target_dataset] = deep_update(
            config_base,
            config_all.get(target_dataset, {})
        )
    return return_config


def flatten_dict(in_dict: dict, k_parent: str = ''):
    out_dict = {}
    for k in in_dict:
        out_key = k if k_parent == '' else f"{k_parent}/{k}"
        if isinstance(in_dict[k], dict):
            out_dict = flatten_dict(in_dict[k], out_key)
        else:
            out_dict[out_key] = in_dict[k]
    return out_dict


def build_search_space(in_dict: dict, sample_function="grid_search"):

    out_dict = {}
    for k in in_dict:
        if isinstance(in_dict[k], dict):
            out_dict[k] = build_search_space(in_dict[k], sample_function)
        elif isinstance(in_dict[k], list):
            out_dict[k] = getattr(tune, sample_function)(in_dict[k])
        elif isinstance(in_dict[k], (Union[int, float, str]).__args__):
            out_dict[k] = getattr(tune, sample_function)([in_dict[k]])
        else:
            raise TypeError(f"what to do with type: {type(in_dict[k])}")
    return out_dict
