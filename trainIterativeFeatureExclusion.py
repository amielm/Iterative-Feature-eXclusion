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

from scipy.stats import entropy

from xgboost import XGBClassifier
from xgboost.callback import EarlyStopping


from IterativeFeatureExclusion import IterativeFeatureExclusion
from utils import datasets_list, calculate_scores, hyperopt_XGBoost

global_random_state = 42

def train_iterative_feature_exclusion(X_train, y_train, X_test, y_test, fold_id, repetition, d, experiment_time_tag):
    wandb_config = {"repetition": repetition,
                    "fold_id": fold_id,
                    "dataset_name": d,
                    "n_features": X_train.shape[1],
                    }

    run = wandb.init(project=f"weak feature boosting",
                     name=f"{d}",
                     job_type='demonstration',
                     notes="",
                     settings=wandb.Settings(start_method="thread"),
                     config=wandb_config,
                     tags=["DART", f"time:{experiment_time_tag:.0f}"],
                     )

    metrics_dict = {}

    # list of baseline models
    simple_classifiers_list = [
        ('NB',make_pipeline(StandardScaler(),
                        GaussianNB())),
        ('LR',LogisticRegression(n_jobs=-1, random_state=global_random_state)),
        ('SVM', make_pipeline(StandardScaler(),
                              LinearSVC(random_state=global_random_state))),
        ('DT', DecisionTreeClassifier(random_state=global_random_state)),
        ('RF', RandomForestClassifier(n_jobs=-1, n_estimators=250, random_state=global_random_state))
    ]

    # train and evaluate models on raw data 4 base models + RF and XGBoost
    for model_name, model_pipeline in simple_classifiers_list:
        base_model = clone(model_pipeline)
        base_model.fit(X_train, y_train)
        base_model_y_pred_train, base_model_y_pred_test = base_model.predict(X_train), base_model.predict(X_test)
        if model_name == "SVM":
            base_model_y_scores_train, base_model_y_scores_test = base_model.predict(X_train), base_model.predict(X_test)
        else:
            base_model_y_scores_train, base_model_y_scores_test = base_model.predict_proba(X_train), base_model.predict_proba(X_test)
        metrics_dict.update(calculate_scores(f"train_{model_name}_raw_data", base_model_y_pred_train, base_model_y_scores_train, y_train, y_train))
        metrics_dict.update(calculate_scores(f"test_{model_name}_raw_data", base_model_y_pred_test, base_model_y_scores_test, y_test, y_train))


    # hyper-opt (10 iters), train and evaluate XGBoost
    XGBoost_best_params = hyperopt_XGBoost(X_train, y_train, random_state=18, max_evals=10)
    # train XGBoost (with random_seed)
    es = EarlyStopping(rounds=50, min_delta=0, save_best=True, maximize=True, data_name="validation_1", metric_name="roc_auc_score")
    base_model = XGBClassifier(eval_metric=roc_auc_score, n_jobs=-1, random_state=global_random_state, callbacks=[es], booster="dart", **XGBoost_best_params)
    X_train_xgb_base, X_val_xgb_base, y_train_xgb_base, y_val_xgb_base = train_test_split(X_train, y_train,
                                                                                          test_size=0.2,
                                                                                          stratify=y_train,
                                                                                          random_state=17)
    # the second eval_set is used for early stopping
    base_model.fit(X_train_xgb_base, y_train_xgb_base, eval_set=[(X_train_xgb_base, y_train_xgb_base), (X_val_xgb_base, y_val_xgb_base)])
    base_model_y_pred_train, base_model_y_pred_test = base_model.predict(X_train), base_model.predict(X_test)
    base_model_y_scores_train, base_model_y_scores_test = base_model.predict_proba(
        X_train), base_model.predict_proba(
        X_test)
    metrics_dict.update(
        calculate_scores(f"train_XGB_raw_data", base_model_y_pred_train, base_model_y_scores_train, y_train,
                         y_train))
    metrics_dict.update(
        calculate_scores(f"test_XGB_raw_data", base_model_y_pred_test, base_model_y_scores_test, y_test,
                         y_train))

    # iterative feature exclusion
    n_iter = 10
    one_feature_drop_frac = 1 / (X_train.shape[1] + 1)
    iterative_booster_params = {
        "n_iter": n_iter,
        "frac_drop": one_feature_drop_frac,
        "ensemble_type": "boosting",
        "random_state": global_random_state}
    iterative_booster_model = IterativeFeatureExclusion(XGBoost_init_params=XGBoost_best_params,
                                                        **iterative_booster_params)
    iterative_booster_model.fit(X_train, y_train)
    iterative_booster_model_y_pred_train, iterative_booster_model_y_pred_test = iterative_booster_model.predict(X_train), iterative_booster_model.predict(X_test)
    iterative_booster_model_y_scores_train, iterative_booster_model_y_scores_test = iterative_booster_model.predict_proba(
        X_train), iterative_booster_model.predict_proba(
        X_test)
    metrics_dict.update(
        calculate_scores(f"train_weak_boosting_raw_data", iterative_booster_model_y_pred_train, iterative_booster_model_y_scores_train, y_train,
                         y_train))
    metrics_dict.update(
        calculate_scores(f"test_weak_boosting_raw_data", iterative_booster_model_y_pred_test, iterative_booster_model_y_scores_test, y_test,
                         y_train))

    metrics_dict.update({f"xgb_top10_features_shap_entropy": entropy(sorted(iterative_booster_model.feature_importance_all_iters[0])[-10:])})


    # log predictions after each feature removal round
    for i in range(n_iter):
        iter_metrics_dict = {}
        prediction_iter_for_round_i = iterative_booster_model.weak_boosting_iters_info[i]["xgb_prediction_iteration"]
        iterative_booster_model_y_pred_train = iterative_booster_model.predict_weak_boosting_iter_i(X_train, iteration=prediction_iter_for_round_i)
        iterative_booster_model_y_pred_test = iterative_booster_model.predict_weak_boosting_iter_i(X_test, iteration=prediction_iter_for_round_i)
        iterative_booster_model_y_scores_train = iterative_booster_model.predict_proba_weak_boosting_iter_i(
            X_train, iteration=prediction_iter_for_round_i)
        iterative_booster_model_y_scores_test = iterative_booster_model.predict_proba_weak_boosting_iter_i(
            X_test, iteration=prediction_iter_for_round_i)
        iter_metrics_dict.update(
            calculate_scores(f"train_weak_boosting_raw_data_per_iter", iterative_booster_model_y_pred_train,
                             iterative_booster_model_y_scores_train, y_train, y_train))
        iter_metrics_dict.update(
            calculate_scores(f"test_weak_boosting_raw_data_per_iter", iterative_booster_model_y_pred_test,
                             iterative_booster_model_y_scores_test, y_test, y_train))
        wandb.log(iter_metrics_dict)

    metrics_dict.update({"weak_boosting_iterations_info": iterative_booster_model.weak_boosting_iters_info})

    wandb.log(metrics_dict)
    wandb.finish()
    return

base_path = r"datasets/"

# env variables only relevant if using wandb and running on slurm cluster
os.environ["WANDB_API_KEY"] = "my_key"
os.environ["WANDB_DISABLE_SERVICE"] = "True"
#os.environ["WANDB_MODE"] = "online"
os.environ["WANDB_MODE"] = "offline"
#os.environ["WANDB_MODE"] = "disabled"
slurm_task_id = int(os.environ["SLURM_ARRAY_TASK_ID"])
# cur_dataset_ind = slurm_task_id
experiment_time_tag = float(os.environ["SLURM_ARRAY_JOB_ID"])

n_folds = 10
n_repetition = 10
n_datasets = len(datasets_list)

# get the current id numbers for dataset, fold, and repetition
cur_dataset_ind = int(slurm_task_id % n_datasets)
remaining_id = np.floor(slurm_task_id / n_datasets)
fold_id = int(remaining_id % n_folds)
remaining_id = np.floor(remaining_id / n_folds)
repetition = int(remaining_id)

d = datasets_list[cur_dataset_ind]
print(f"Fold:{fold_id}, Repetition:{repetition}, Dataset: {cur_dataset_ind} - {d}")
file_name = join(base_path, d)
df = pd.read_csv(file_name)
# The last column has the correct labels (targets)
# All other are input features
X, y = df.iloc[:, :df.shape[1] - 1].values, df.iloc[:, df.shape[1] - 1].values


state = global_random_state + repetition
skf = StratifiedKFold(n_splits=n_folds, random_state=state, shuffle=True)
train, test = list(skf.split(X, y))[fold_id]
X_train, X_test, y_train, y_test = X[train], X[test], y[train], y[test]

train_iterative_feature_exclusion(X_train, y_train, X_test, y_test, fold_id, repetition, d,
                                  experiment_time_tag)