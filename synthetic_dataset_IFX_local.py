import sys
import wandb

import numpy as np
import pandas as pd
from os.path import join
from tqdm import tqdm
import time
import os
import re
import json

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.ensemble import StackingClassifier, RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.base import clone
from sklearn.metrics import roc_auc_score

from xgboost import XGBClassifier
from xgboost.callback import EarlyStopping

# from matplotlib import pyplot as plt
# import seaborn as sns

from IterativeWeakBooster import IterativeWeakBoosterXGBoost
from utils import dataset_list, bigger_datasets_list, calculate_scores, hyperopt_XGBoost, create_synthetic_dataset, create_synthetic_xor

global_random_state = 17


def train_IFX_synthetic(X_train, y_train, X_test, y_test, fold_id, repetition, experiment_time_tag, n_strong_features, target_correlation, n_iter, verbose=1, metric="auc"):

    metrics_dict = {}

    # hyper-opt (5-10 iters), train and evaluate XGBoost
    hyperopt_start_time = time.time()
    XGBoost_best_params = hyperopt_XGBoost(X_train, y_train, random_state=18, max_evals=hyperopt_max_evals, verbose=verbose)
    hyperopt_time = time.time() - hyperopt_start_time
    metrics_dict.update({"XGB_hyperopt_time": hyperopt_time})

    # weak "boosting" on raw data
    one_feature_drop_frac = 1 / (X_train.shape[1] + 1)
    iterative_booster_params = {
        "n_iter": n_iter,
        "frac_drop": one_feature_drop_frac,
        "ensemble_type": "boosting",
        "random_state": global_random_state,
        "verbose": 0}
    iterative_booster_model = IterativeWeakBoosterXGBoost(XGBoost_init_params=XGBoost_best_params,
                                                          **iterative_booster_params)
    start_IFX_training_time = time.time()
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


    if metric == "auc":
        final_xgb_performance = metrics_dict['test_weak_boosting_raw_data_iter_0_roc_auc']
        final_ifx_performance = metrics_dict[f'test_weak_boosting_raw_data_iter_{n_iter-1}_roc_auc']
    elif metric == "accuracy":
        final_xgb_performance = metrics_dict['test_weak_boosting_raw_data_iter_0_accuracy']
        final_ifx_performance = metrics_dict[f'test_weak_boosting_raw_data_iter_{n_iter - 1}_accuracy']

    if verbose>0:
        print(f"Final performance comparison: XGBoost: {final_xgb_performance} IFX: {final_ifx_performance}")


    return final_xgb_performance, final_ifx_performance


# setup
experiment_time_tag = time.time()


n_repetitions=20
n_noise_features= -1
target_correlation= -1
n_splits = 10
n_iter = 5
hyperopt_max_evals = 10
n_features = 20


# some of the sample sizes, running all of them takes a long time
sample_sizes = [96, 192, 384, 768, 1024]
strong_feature_thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
n_strong_features = 1

results = []
for threshold_id, threshold in tqdm(enumerate(strong_feature_thresholds[::-1])):
    for repetition in tqdm(range(n_repetitions), disable=True):
        state = 13 + (repetition*100) + round(threshold_id*10)

        X_train_all, y_train_all = create_synthetic_xor(n_features=n_features, n_samples=np.max(sample_sizes),
                                                        strong_feature_threshold=threshold, random_seed=state)
        X_test, y_test = create_synthetic_xor(n_features=n_features, n_samples=np.max(sample_sizes),
                                              strong_feature_threshold=threshold, random_seed=2*state)


        repetition_results = []
        for n_samples in sample_sizes:
            X_train, y_train = X_train_all[:n_samples, :], y_train_all[:n_samples]

            model = XGBClassifier(
                    eval_metric=roc_auc_score,
                    n_jobs=-1,
                    random_state=state)
            model.fit(X_train, y_train)
            y_test_pred = model.predict_proba(X_test)[:, 1]
            # print(roc_auc_score(y_test, y_test_pred))

            fold_id=0
            xgb_auc, ifx_auc = train_IFX_synthetic(X_train, y_train, X_test, y_test, fold_id, repetition,
                                                   experiment_time_tag, n_strong_features, target_correlation, n_iter,
                                                   verbose=0)

            current_results = {"xgb_auc": xgb_auc,
                               "ifx_auc": ifx_auc,
                               "n_samples": n_samples,
                               "threshold": threshold}
            repetition_results.append(current_results)
            print(current_results)

        results = results + repetition_results


base_output_path = r"base/path/for/results"
with open(join(base_output_path, f"output_file.json"), "w") as f:
    json.dump(results, f)