import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from xgboost import XGBRFClassifier, XGBClassifier, DMatrix
from xgboost.callback import EarlyStopping
# from utils import CustomEarlyStopping as EarlyStopping
from shap import TreeExplainer
from tqdm import tqdm

class IterativeFeatureExclusion(BaseEstimator, ClassifierMixin):

    def __init__(self, XGBoost_init_params={}, n_estimators_per_iter=10, n_iter=10, frac_drop=0.05, random_state=17, ensemble_type="boosting"):
        """
        :param n_estimators_per_iter: number of trees to learn in each iteration
        :param n_iter: number of iterations of dropping strong features
        :param frac_drop: what percentage of features to drop in each iteration
        """

        self.n_estimators_per_iter = n_estimators_per_iter
        self.n_iter = n_iter
        self.frac_drop = frac_drop
        self.XGBoost_init_params = XGBoost_init_params
        self.random_state = random_state
        self.ensemble_type = ensemble_type
        self.feature_importance_all_iters = []
        self.weak_boosting_iters_info = []

        if ensemble_type == "boosting":
            self.model, self.es = self.init_xgb()

    def init_xgb(self):
        es = EarlyStopping(
            rounds=50,
            min_delta=0,
            save_best=True,
            maximize=True,
            data_name="validation_1",
            metric_name="roc_auc_score",
        )

        model = XGBClassifier(
            # booster="gbtree",
            booster="dart",
            eta=0.1,
            importance_type="weight",
            random_state=self.random_state,
            n_jobs=-1,
            eval_metric=roc_auc_score,
            callbacks=[es],
            **self.XGBoost_init_params)

        return model, es

    def fit(self, X, y):
        # Check that X and y have correct shape
        X, y = check_X_y(X, y, force_all_finite="allow-nan")
        # Store the classes seen during fit
        self.classes_ = unique_labels(y)

        # siphon off validation data
        X, X_val, y, y_val = train_test_split(X, y, test_size=0.2, stratify=y,
                                                          random_state=17)

        self.X_ = X.copy()
        self.y_ = y.copy()
        self.n_features = X.shape[1]
        self.n_features_to_drop = int(np.ceil(self.n_features * self.frac_drop))

        # initialize the masks use select the features for each iteration
        self.feature_masks = np.ones((self.n_iter, self.n_features), dtype=bool)

        # train first iter on unfiltered data
        cur_X = X.copy()
        self.model.fit(cur_X, y, eval_set=[(cur_X, y), (X_val, y_val)])
        # condition taken from the early stopping callback stopping condition
        if self.es.current_rounds >= self.es.rounds:
            print(
                f"Early stopping triggered at iteration {self.model.best_iteration}")
        # if early stopping not activated then the learning never reached a plateau
        else:
            print(f"Early stopping not triggered")

        # log relevant info about the baseline training
        self.weak_boosting_iters_info.append({"weak_boosting_iteration": 0,
                                              "xgb_prediction_iteration": self.model.best_iteration,
                                              "features_removed_before_iteration": []})

        self.prev_best_iter = 0
        feature_importances = []
        for i in tqdm(range(1, self.n_iter)):
            if self.n_features_to_drop > 0:
                # get feature importance for the features.
                current_iter_feature_importance_ranks = self.get_feature_importance(self.model, method="shap",
                                                                                    cur_X=cur_X)

                drop_new_features = True
                if drop_new_features:
                    # next few lines are to ensure that features that were already dropped are not re-selected
                    dropped_features = np.where(np.logical_not(self.feature_masks[i - 1, :]))
                    dropped_features_mask = np.isin(current_iter_feature_importance_ranks, dropped_features)
                    dropable_ranked_features = current_iter_feature_importance_ranks[~dropped_features_mask]
                else:
                    # can "re-drop" features if it is still the most influential.
                    dropable_ranked_features = current_iter_feature_importance_ranks

                feature_inds_to_drop = dropable_ranked_features[:self.n_features_to_drop]
                print(f"dropped features : {feature_inds_to_drop}")
                feature_importances.append(current_iter_feature_importance_ranks)
                # drop for all subsequent iterations
                self.feature_masks[i:, feature_inds_to_drop] = False

                # reset data so features can come back in this setup if their importance drops after a few iterations
                cur_X = X.copy()
                # zero out columns instead of not extracting them. This makes splitting on them impossible.
                cur_X[:, np.logical_not(self.feature_masks[i, :])] = 0

                # fit the current XGBoost model.
                new_model, new_es = self.init_xgb()
                self.prev_best_iter = self.model.best_iteration
                new_model.fit(cur_X, y, xgb_model=self.model.get_booster(), eval_set=[(cur_X, y), (X_val, y_val)])
                # ensure that the new_model actually improves on the existing model
                prev_best_score = self.es.best_scores[self.es.data][self.es.metric_name][-1]
                current_best_score = new_es.best_scores[new_es.data][new_es.metric_name][-1]
                if current_best_score > prev_best_score:
                # if new_model.best_iteration - 1 > self.prev_best_iter:
                    self.model, self.es = new_model, new_es
                    # condition taken from the early stopping callback stopping condition
                    if self.es.current_rounds >= self.es.rounds:
                        print(
                            f"Early stopping triggered at xgboost iteration {self.model.best_iteration - self.prev_best_iter - 1, self.model.best_iteration}")
                    # if early stopping not activated then the learning never reached a plateau
                    else:
                        print(f"Early stopping not triggered at iteration {i}")
                else:
                    print(f"no improvement in iteration {i}")

                # log relevant info the current iteration of weak boosting
                self.weak_boosting_iters_info.append({"weak_boosting_iteration": i,
                                                      "xgb_prediction_iteration": self.model.best_iteration,
                                                      "features_removed_before_iteration": feature_inds_to_drop})


        _ = self.get_feature_importance(self.model, method="shap",
                                      cur_X=cur_X)
        self.best_iter = self.model.best_iteration
        return self

    def get_feature_importance(self, model, method="rf_importance", cur_X=None):
        """
        Get the feature importance sorted from most to least important feature
        :param model: the RF to extract the importance from
        :param method: how to perform the feature importance calculation
        :return: the feature importance sorted from most to least important
        """
        if method=="rf_importance":
            # use the feature importance built into
            current_feature_importance = model.feature_importances_

        elif method=="shap":
            # Using SHAP package
            # explainer = TreeExplainer(model, model_output="raw")
            # shap_values = explainer.shap_values(cur_X, check_additivity=False)

            # built-in method for calculating shap values - supports DART
            shap_values = model.get_booster().predict(DMatrix(cur_X), pred_contribs=True)[:,:-1]

            # calculate feature importance as the mean abs of the shap value per feature
            global_shap_values = np.abs(shap_values).sum(axis=0)
            current_feature_importance = global_shap_values

        # accumulate feature importances for analysis
        self.feature_importance_all_iters.append(current_feature_importance)

        return current_feature_importance.argsort()[::-1]

    def predict(self, X):
        return self.predict_proba(X).argmax(axis=1)

    def predict_proba(self, X):
        # Check if fit has been called
        check_is_fitted(self)

        # Input validation
        # X = check_array(X)

        predictions = self.predict_proba_all_iterations(X)[-1]

        return predictions


    def predict_proba_weak_boosting_iter_i(self, X, iteration):
        return self.model.predict_proba(X, iteration_range=(0, iteration+1))

    def predict_weak_boosting_iter_i(self, X, iteration):
        return self.predict_proba_weak_boosting_iter_i(X, iteration).argmax(axis=1)

    def predict_proba_all_iterations(self, X):
        predictions = [self.model.predict_proba(X, iteration_range=(0,i+1)) for i in
                       range(self.model.best_iteration+1)]
        return predictions
