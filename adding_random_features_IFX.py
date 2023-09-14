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


def train_IFX(X_train, y_train, X_test, y_test, fold_id, repetition, d, experiment_time_tag, n_noise_features):

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

    with open(join(current_results_path, f"{d}_fold{fold_id}_rep{repetition}_n_noise{n_noise_features}.json"), "w") as f:
        json.dump(metrics_dict, f)

    return


# setup
slurm_task_id = int(os.environ["SLURM_ARRAY_TASK_ID"])
experiment_time_tag = float(os.environ["SLURM_JOBID"])

base_path = r"my/base/path"
base_results_path = "base/path/for/results"
current_results_path = join(base_results_path, "adding_noise_to_datasets", "10x10_full_experiment")
while not os.path.exists(current_results_path):
    if slurm_task_id == 0:
        os.makedirs(current_results_path)
    else:
        time.sleep(1)

# hyper-parameters
n_folds = 10
n_repetition = 10
n_datasets = len(bigger_datasets_list)
n_noise_features_range = [1,2,4,8,16,32,64,128]
hyperopt_max_evals = 10
n_iter = 10


# get the current id numbers for dataset, fold, and repetition
cur_dataset_ind = int(slurm_task_id % n_datasets)
remaining_id = np.floor(slurm_task_id / n_datasets)
n_noise_features_ind = int(remaining_id % len(n_noise_features_range))
remaining_id = np.floor(remaining_id / len(n_noise_features_range))
repetition = int(remaining_id % n_repetition)

d = bigger_datasets_list[cur_dataset_ind]
n_noise_features = n_noise_features_range[n_noise_features_ind]
print(f"Repetition:{repetition}, #Noise Features {n_noise_features}, Dataset: {cur_dataset_ind} - {d}")

state = global_random_state + repetition + 10*cur_dataset_ind + 1000*n_noise_features
rng = np.random.default_rng(seed=state)

file_name = join(base_path, d)
df = pd.read_csv(file_name)
# Assume: The last column has the correct labels (targets) All other are input features
X, y = df.iloc[:, :df.shape[1] - 1].values, df.iloc[:, df.shape[1] - 1].values

# Adding noise features
noise_features = rng.uniform(low=0., high=1., size=(X.shape[0], n_noise_features))
X = np.hstack((noise_features, X))

skf = StratifiedKFold(n_splits=n_folds, random_state=state, shuffle=True)
for fold_id in range(n_folds):
    print(f"--------fold {fold_id}------------")
    train, test = list(skf.split(X, y))[fold_id]
    X_train, X_test, y_train, y_test = X[train], X[test], y[train], y[test]

    train_IFX(X_train, y_train, X_test, y_test, fold_id, repetition, d, experiment_time_tag, n_noise_features)
