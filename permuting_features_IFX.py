# import wandb

import sys
import numpy as np
import pandas as pd
from os.path import join
from tqdm import tqdm
import time
import os
import re
import json

from sklearn.model_selection import StratifiedKFold, train_test_split

from xgboost import XGBClassifier
from xgboost.callback import EarlyStopping

from IterativeWeakBooster import IterativeWeakBoosterXGBoost
from utils import dataset_list, bigger_datasets_list, calculate_scores, hyperopt_XGBoost, create_synthetic_dataset

global_random_state = 17


def train_IFX(X_train, y_train, X_test, y_test, fold_id, repetition, d, experiment_time_tag, permutation_fraction):

    metrics_dict = {}

    # hyper-opt (5-10 iters), train and evaluate XGBoost
    hyperopt_start_time = time.time()
    XGBoost_best_params = hyperopt_XGBoost(X_train, y_train, random_state=18, max_evals=hyperopt_max_evals)
    hyperopt_time = time.time() - hyperopt_start_time
    metrics_dict.update({"XGB_hyperopt_time": hyperopt_time})

    # weak "boosting" on raw data
    one_feature_drop_frac = 1 / (X_train.shape[1] + 1)
    iterative_booster_params = {
        "n_iter": n_iter,
        "frac_drop": one_feature_drop_frac,
        "ensemble_type": "boosting",
        "random_state": global_random_state}
    iterative_booster_model = IterativeWeakBoosterXGBoost(XGBoost_init_params=XGBoost_best_params,
                                                          **iterative_booster_params)
    start_IFX_training_time = time.time()
    metrics_dict.update({"start_IFX_training_time":start_IFX_training_time})
    iterative_booster_model.fit(X_train, y_train)
    iterative_booster_model_y_pred_train, iterative_booster_model_y_pred_test = iterative_booster_model.predict(
        X_train), iterative_booster_model.predict(X_test)
    iterative_booster_model_y_scores_train, iterative_booster_model_y_scores_test = iterative_booster_model.predict_proba(
        X_train), iterative_booster_model.predict_proba(
        X_test)
    metrics_dict.update(
        calculate_scores(f"train_weak_boosting_raw_data", iterative_booster_model_y_pred_train,
                         iterative_booster_model_y_scores_train, y_train,
                         y_train))
    metrics_dict.update(
        calculate_scores(f"test_weak_boosting_raw_data", iterative_booster_model_y_pred_test,
                         iterative_booster_model_y_scores_test, y_test,
                         y_train))

    # log predictions after each feature removal round
    for i in range(n_iter):
        prediction_iter_for_round_i = iterative_booster_model.weak_boosting_iters_info[i]["xgb_prediction_iteration"]
        iterative_booster_model_y_pred_train = iterative_booster_model.predict_weak_boosting_iter_i(X_train,
                                                                                                    iteration=prediction_iter_for_round_i)
        iterative_booster_model_y_pred_test = iterative_booster_model.predict_weak_boosting_iter_i(X_test,
                                                                                                   iteration=prediction_iter_for_round_i)
        iterative_booster_model_y_scores_train = iterative_booster_model.predict_proba_weak_boosting_iter_i(
            X_train, iteration=prediction_iter_for_round_i)
        iterative_booster_model_y_scores_test = iterative_booster_model.predict_proba_weak_boosting_iter_i(
            X_test, iteration=prediction_iter_for_round_i)
        metrics_dict.update(
            calculate_scores(f"train_weak_boosting_raw_data_iter_{i}", iterative_booster_model_y_pred_train,
                             iterative_booster_model_y_scores_train, y_train, y_train))
        metrics_dict.update(
            calculate_scores(f"test_weak_boosting_raw_data_iter_{i}", iterative_booster_model_y_pred_test,
                             iterative_booster_model_y_scores_test, y_test, y_train))

    metrics_dict.update({"XGB_time": iterative_booster_model.weak_boosting_iters_info[0]["timestamp"] - start_IFX_training_time,
                         "IFX_time": iterative_booster_model.weak_boosting_iters_info[-1]["timestamp"] - start_IFX_training_time,
                         })

    metrics_dict.update({"iterations_meta_data": iterative_booster_model.weak_boosting_iters_info})

    # wandb.log(metrics_dict)
    # wandb.finish()

    with open(join(current_results_path, f"{d}_fold{fold_id}_rep{repetition}_peprmutation-fraction{permutation_fraction}.json"), "w") as f:
        json.dump(metrics_dict, f)

    return


def permute_fraction_of_values(arr, fraction, rng):
    """
    Permute a fraction of the values in each column of a NumPy array.

    Parameters:
        arr (numpy.ndarray): The input NumPy array.
        fraction (float): The fraction of values to permute in each column. Should be between 0 and 1.

    Returns:
        numpy.ndarray: A new NumPy array with the permuted values.
    """
    if not 0 <= fraction <= 1:
        raise ValueError("Fraction should be between 0 and 1.")

    num_rows, num_cols = arr.shape
    num_to_permute = int(num_rows * fraction)

    permuted_arr = np.copy(arr)

    for col in range(num_cols):
        # Get the indices of the fraction of values to permute for the current column
        indices_to_permute = rng.choice(num_rows, num_to_permute, replace=False)

        # Permute the selected values
        permuted_arr[indices_to_permute, col] = rng.permutation(permuted_arr[indices_to_permute, col])

    return permuted_arr


# setup
slurm_task_id = int(os.environ["SLURM_ARRAY_TASK_ID"])
experiment_time_tag = float(os.environ["SLURM_JOBID"])

base_path = r"my/base/path"
base_results_path = "base/path/for/results"
current_results_path = join(base_results_path, "permuting_features", "10x10_full_experiment")
while not os.path.exists(current_results_path):
    if slurm_task_id == 0:
        os.makedirs(current_results_path)
    else:
        time.sleep(1)

# hyper-parameters
n_folds = 10
n_repetition = 10
n_datasets = len(bigger_datasets_list)
permutation_fraction_range = [0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40]
hyperopt_max_evals = 10
n_iter = 10


# get the current id numbers for dataset, fold, and repetition
cur_dataset_ind = int(slurm_task_id % n_datasets)
remaining_id = np.floor(slurm_task_id / n_datasets)
permutation_fraction_ind = int(remaining_id % len(permutation_fraction_range))
remaining_id = np.floor(remaining_id / len(permutation_fraction_range))
repetition = int(remaining_id % n_repetition)

d = bigger_datasets_list[cur_dataset_ind]
permutation_fraction = permutation_fraction_range[permutation_fraction_ind]
print(f"Repetition:{repetition}, %Permutation {permutation_fraction}, Dataset: {cur_dataset_ind} - {d}")

state = global_random_state + repetition + 10 * cur_dataset_ind + 1000 * int(100*permutation_fraction)
rng = np.random.default_rng(seed=state)

file_name = join(base_path, d)
df = pd.read_csv(file_name)
# Assume: The last column has the correct labels (targets) All other are input features
X, y = df.iloc[:, :df.shape[1] - 1].values, df.iloc[:, df.shape[1] - 1].values

# Permuting permutation_fraction of each feature
X = permute_fraction_of_values(X, permutation_fraction, rng)

skf = StratifiedKFold(n_splits=n_folds, random_state=state, shuffle=True)
for fold_id in range(n_folds):
    print(f"--------fold {fold_id}------------")
    train, test = list(skf.split(X, y))[fold_id]
    X_train, X_test, y_train, y_test = X[train], X[test], y[train], y[test]

    train_IFX(X_train, y_train, X_test, y_test, fold_id, repetition, d, experiment_time_tag, permutation_fraction)
