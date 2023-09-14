import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve, auc, precision_recall_curve, roc_auc_score

from sklearn.model_selection import train_test_split

from xgboost.callback import EarlyStopping

import wandb

datasets_list = ['Kaggle_Surgical-deepnet.csv', 'AcousticExtinguisherFire.csv', 'chess-krvkp.csv', 'kaggle_REWEMA.csv',
                        'madelon.csv', 'OPENML_philippine.csv', 'ozone.csv', 'Pistachio_28_Features_Dataset.csv', 'seismic-bumps.csv',
                        'spambase.csv']


def calculate_scores(prefix, y_pred, y_score, y_true, y_train, num_samples=True):
    """
    Calculate scoring metrics for binary classification datasets
    :param prefix: A prefix to be added to all the keys in the returned in the result dictionary
    :param y_pred: class predictions
    :param y_score: class probabilities
    :param y_true: ground truth classification
    :param y_train: training ground truth classification (used for detecting majority class)
    :return: dictionary with the results metrics
    """
    y_score = np.squeeze(y_score)
    if len(y_score.shape)==2:
        y_score_0, y_score_1 = y_score[:,0], y_score[:,1]
    elif len(y_score.shape)==1:
        y_score_0, y_score_1 = 1-y_score, y_score

    # calculate performance metrics
    accuracy = accuracy_score(y_true, y_pred)
    fpr, tpr , _ = roc_curve(y_true, y_score_1)
    roc_auc = auc(fpr, tpr)
    
    #     first calculate precision and recall curve for class 1 then 0
    precision_class_0, recall_class_0, thresholds_class_0 = precision_recall_curve(1-y_true, y_score_0)
    precision_class_1, recall_class_1, thresholds_class_1 = precision_recall_curve(y_true, y_score_1)
    #     then assign majority/minority prefix
    majority_class_index = pd.value_counts(y_train).keys()[0]
    if majority_class_index == 0:
        majority_pr_auc = auc(recall_class_0, precision_class_0)
        minority_pr_auc = auc(recall_class_1, precision_class_1)
    else:
        majority_pr_auc = auc(recall_class_1, precision_class_1)
        minority_pr_auc = auc(recall_class_0, precision_class_0)
    
#     expected_predictions = np.average(y_pred, axis=0)
#     bias_squared_predictions = expected_bias_squared(expected_predictions, y_true)
#     variance_predictions = expected_variance(y_pred, expected_predictions)

#     expected_score = np.average(y_score[:,1], axis=0)
#     bias_squared_score = expected_bias_squared(expected_score, y_true)
#     variance_score = expected_variance(y_score[:,1], expected_score)

    results_dict = {  
        "accuracy": accuracy,
        "roc_auc": roc_auc,
        "majority_pr_auc": majority_pr_auc,
        "minority_pr_auc": minority_pr_auc,
        # "bias_squared_predictions": bias_squared_predictions,
        # "variance_predictions": variance_predictions,
        # "bias_squared_score": bias_squared_score,
        # "variance_score": variance_score,
    }

    # add number of samples for calculating the averages
    if num_samples:
        results_dict["num_samples"] = y_pred.shape[0]


    results_dict = {f"{prefix}_{key}": val for key, val in results_dict.items()}
    return results_dict

def hyperopt_XGBoost(X, y, random_state=17, max_evals=1000):
    from hyperopt import fmin, tpe, hp
    from xgboost import XGBClassifier
    from xgboost.callback import EarlyStopping
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import roc_auc_score

    # Define the search space for the hyperparameters
    space = {
        'n_estimators': hp.quniform('n_estimators', 100, 4000, 1),
        'learning_rate': hp.loguniform('learning_rate', -7, 0),
        'max_depth': hp.quniform('max_depth', 1, 10, 1),
        'subsample': hp.uniform('subsample', 0.2, 1),
        'colsample_bytree': hp.uniform('colsample_bytree', 0.2, 1),
        'colsample_bylevel': hp.uniform('colsample_bylevel', 0.2, 1),
        'min_child_weight': hp.loguniform('min_child_weight', -16, 5),
        'reg_lambda':  hp.choice('reg_lambda', [0, hp.loguniform('lambda_log', -16, 2)]),
        'reg_alpha': hp.choice('reg_alpha', [0, hp.loguniform('alpha_log', -16, 2)]),
        'gamma': hp.choice('gamma', [0, hp.loguniform('gamma_log', -16, 2)]),
        'rate_drop': hp.uniform('rate_drop', 0, 1)
    }

    # Define the objective function to optimize
    def objective(params):
        repetition_scores = []
        for i in range(5):
            # this is an addition on the article we base the hyperopt on ...
            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=random_state+i)
            es = EarlyStopping(
                rounds=50,
                min_delta=0,
                save_best=True,
                maximize=True,
                data_name="validation_0",
                metric_name="roc_auc_score",
            )
            model = XGBClassifier(
                n_estimators=int(params['n_estimators']),
                learning_rate=params['learning_rate'],
                max_depth=int(params['max_depth']),
                subsample=params['subsample'],
                colsample_bytree=params['colsample_bytree'],
                colsample_bylevel=params['colsample_bylevel'],
                min_child_weight=params['min_child_weight'],
                reg_lambda=params['reg_lambda'],
                reg_alpha=params['reg_alpha'],
                gamma=params['gamma'],
                eval_metric=roc_auc_score,
                n_jobs=-1,
                random_state=random_state,
                booster="dart",
                rate_drop = params['rate_drop'],
                callbacks=[es]
            )
            model.fit(X_train, y_train, eval_set=[(X_val,y_val)], verbose=0)
            y_val_pred = model.predict_proba(X_val)[:, 1]
            repetition_score = roc_auc_score(y_val, y_val_pred)

            repetition_scores.append(repetition_score)

            # negative value because hyperopt minimizes the objective...
        return -np.mean(repetition_scores)

    rstate = np.random.default_rng(random_state)
    best_params = fmin(objective, space, algo=tpe.suggest, max_evals=max_evals, rstate=rstate)


    best_params["reg_lambda"] = best_params.pop("lambda_log",  best_params["reg_lambda"])
    best_params["reg_alpha"] = best_params.pop("alpha_log",  best_params["reg_alpha"])
    best_params["gamma"] = best_params.pop("gamma_log",  best_params["gamma"])
    best_params["n_estimators"] = int(best_params["n_estimators"])
    best_params["max_depth"] = int(best_params["max_depth"])

    # Print the best parameters
    print(f"Optimal params: {best_params}")

    return best_params


# change the best_iteration attribute so SHAP uses all of the models until the current iteration
class CustomEarlyStopping(EarlyStopping):
    def after_training(self, model):
        if self.current_rounds == 0:
            self.actual_best_iteration = int(model.attr("best_iteration")) + 1
        # use a very high number that way all the iterations learnt so far are used even if they are not optimal
        model.set_attr(best_iteration=str(100000))
        return model

def create_synthetic_xor(n_predictive_features=3, n_strong_features=1, n_features=4, n_samples=512,
                         strong_feature_threshold=0.8, random_seed=17):

    if n_features < (n_predictive_features + n_strong_features):
        raise ValueError("number of predictive+strong features can not be more than total number of features")

    rng = np.random.default_rng(seed=random_seed)
    # sample the predictive features as random binary variables
    X_predictive = rng.choice(2, size=(n_samples, n_predictive_features), replace=True)

    # set the target as the XOR ("odd parity")
    odd_parity = np.bitwise_xor.reduce(X_predictive, axis=1)
    y = 1 - odd_parity

    # sample/generate the strong feature(s)
    X_strong = rng.uniform(low=0., high=1., size=(n_samples, n_strong_features))
    for i in range(n_strong_features):
        # uncorrelated_mask = np.logical_and(X_strong[:,i]>0.4,
        #                X_predictive[:,0])
        uncorrelated_mask = X_strong[:, i] > strong_feature_threshold
        X_strong[uncorrelated_mask, i] = rng.choice(2, size=uncorrelated_mask.sum(), replace=True)
        X_strong[np.invert(uncorrelated_mask), i] = odd_parity[np.invert(uncorrelated_mask)]

    # add the noisy features
    X_noise = rng.choice(2, size=(n_samples, n_features - (n_predictive_features + n_strong_features)), replace=True)

    # stack the features together
    X = np.hstack((X_strong, X_noise, X_predictive))

    return X, y