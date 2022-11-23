import pickle
import json
import numpy as np


def save_pkl(pipeline, path_to_pkl):
    with open(path_to_pkl, 'wb') as f:
        pickle.dump(pipeline, f)


def save_best_params(best_params, path_to_json):
    with open(path_to_json, 'w', encoding='utf-8') as f:
        json.dump(best_params, f, ensure_ascii=False, indent=4)


# def save_log_file()