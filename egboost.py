from __future__ import annotations
from typing import Any
import time
import pandas as pd
import numpy as np
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.metrics import mean_squared_error
from sklearn.utils.validation import check_is_fitted
import sys
import gc
from enum import Enum
sys.path.insert(0, 'xgboost_compiled/python-package')
import xgboost as xgb
from sklearn.model_selection import train_test_split
from itertools import combinations
from sklearn.utils.extmath import softmax

DEBUG_MODE = False


class FeatureTraverse(Enum):
    """ Feature traverse method.
    """
    BEST_FIT = 1
    RANDOM = 2
    CYCLIC = 3
    CYCLIC_RANDOM = 4
    CYCLIC_RANDOM_REVERSE = 5
    CYCLIC_REVERSE = 6


class ExplainableGBMCore(BaseEstimator, RegressorMixin):
    """ Internal use Explainable Boosting.
    """
    def __init__(
            self,
            max_rounds,
            learning_rate,
            random_state,
            early_stopping_rounds,
            early_stopping_tolerance,
            preprocessor,
            preprocessor_inter,
            objective,
            max_leaves,
            reg_l2,
            reg_l1,
            feature_traverse,
            subsample,
            inner_bags,
            monotone_constraints,
            del_booster
    ) -> None:
        self.monotone_constraints = monotone_constraints
        self.del_booster = del_booster
        self.inner_bags = inner_bags
        self.subsample = subsample
        self.feature_traverse = feature_traverse
        self.reg_l2 = reg_l2
        self.reg_l1 = reg_l1
        self.max_leaves = max_leaves
        self.objective = objective
        self.model_type = "regression"
        if 'binary' in self.objective:
            self.model_type = "classification"
        self.early_stopping_rounds = early_stopping_rounds
        self.early_stopping_tolerance = self.tol = early_stopping_tolerance
        self.max_rounds = max_rounds
        self.learning_rate = learning_rate
        self.domains_ = []
        self.outputs_ = []
        self.outputs_inter_ = []
        self.random_state = random_state
        self.interactions = []
        self.n_features_in_ = 0
        self.num_trees = 0
        self.time_train, self.time_extract = 0, 0
        self.preprocessor_ = preprocessor
        self.preprocessor_inter = preprocessor_inter
        self.interactions_chosen = []
        self.booster = None
        self.predict_uni = None
        self.predict_uni_val = None
        self.features_bin_count = np.array([len(x) for x in self.preprocessor_.col_bin_counts_])
        self.features_bin_count_inter = np.array([len(x) for x in self.preprocessor_inter.col_bin_counts_])
        self.main_fitted = False
        self.inter_fitted = False
        self.init_params = 0
        self.index_stage = 0
        self.order = [[], []]
        self.fit_order = [[], []]

    def fit(
            self, dataset, interactions=None) \
            -> ExplainableGBMCore:
        """ Fits model to provided samples.

        Args:
            dataset: Dataset dictionary.
            interactions: Selected interactions.

        Returns:
            Itself.
        """
        if len(self.interactions_chosen) > 0:
            X, X_val = dataset["X_inter"], dataset["X_val_inter"]
        else:
            X, X_val = dataset["X"], dataset["X_val"]
        y, Y_val = dataset["y"], dataset["Y_val"]

        self.n_features_in_ = X.shape[1]
        if isinstance(y, pd.Series):
            y = y.values
        self.init_params = 0  # np.mean(y)

        if len(self.interactions_chosen) == 0:
            self.predict_uni = np.ones(len(y)) * self.init_params
            self.predict_uni_val = np.ones(len(Y_val)) * self.init_params

        if len(self.interactions_chosen) > 0:
            max_bin = np.max(self.features_bin_count_inter)
        else:
            max_bin = np.max(self.features_bin_count)

        if DEBUG_MODE:
            print("shape val", X_val.shape)
            print("shape train", X.shape)

        if len(self.interactions_chosen) > 0:
            interaction_constraints_base = [[inter[0], inter[1]] for inter in self.interactions_chosen]
        else:
            interaction_constraints_base = [[i] for i in range(self.n_features_in_)]
        num_fgroups = len(interaction_constraints_base)
        self.index_stage = 0
        if len(self.interactions_chosen) > 0:
            self.index_stage = 1

        self.order[self.index_stage] = list(range(len(interaction_constraints_base)))
        if self.feature_traverse[self.index_stage] == FeatureTraverse.BEST_FIT:
            colsample_bytree = 1
        elif self.feature_traverse[self.index_stage] == FeatureTraverse.RANDOM:
            colsample_bytree = 0.0001 # need to be < 0.5
        elif self.feature_traverse[self.index_stage] in [FeatureTraverse.CYCLIC, FeatureTraverse.CYCLIC_REVERSE,
                                                         FeatureTraverse.CYCLIC_RANDOM, FeatureTraverse.CYCLIC_RANDOM_REVERSE]:
            colsample_bytree = 0.6 # need to be  0.5 < and <1
            if self.feature_traverse[self.index_stage] in [FeatureTraverse.CYCLIC_RANDOM, FeatureTraverse.CYCLIC_RANDOM_REVERSE]:
                import random
                if self.feature_traverse[self.index_stage] == FeatureTraverse.CYCLIC_RANDOM_REVERSE and self.random_state % 2 == 1:
                    random.seed(self.random_state-1)
                else:
                    random.seed(self.random_state)
                random.shuffle(self.order[self.index_stage])
            if self.feature_traverse[self.index_stage] in [FeatureTraverse.CYCLIC_RANDOM_REVERSE, FeatureTraverse.CYCLIC_REVERSE] \
                    and self.random_state % 2 == 1:
                self.order[self.index_stage] = self.order[self.index_stage][::-1]
            interaction_constraints_base = [interaction_constraints_base[i] for i in self.order[self.index_stage]]
        start = time.time()
        params = {'eta': self.learning_rate,
                  #'single_precision_histogram': True,
                  'max_depth': 3,
                  'objective': self.objective,
                  'max_leaves': self.max_leaves,
                  'tree_method': 'hist',
                  'max_bin': max_bin,
                  'colsample_bytree': colsample_bytree,
                  'colsample_bylevel': 1,
                  'colsample_bynode': 1,
                  'verbosity': 0,
                  'grow_policy': 'lossguide',
                  'min_child_weight': 2,
                  'interaction_constraints': str(interaction_constraints_base),
                  'validate_parameters': False,
                  'seed': self.random_state,
                  'alpha': self.reg_l1[self.index_stage],
                  # Lasso tends to do well if there are a small number of significant parameters  from: https://www.datacamp.com/tutorial/tutorial-ridge-lasso-elastic-net
                  'lambda': self.reg_l2[self.index_stage],
                  # Ridge works well if there are many large parameters of about the same value
                  'subsample': self.subsample,
                  'num_parallel_tree': self.inner_bags
                  }
        if self.monotone_constraints is not None:
            params["monotone_constraints"] = '(' + ','.join([str(m) for m in self.monotone_constraints]) + ')'

        if self.objective == 'survival:cox':
            params.update({'eval_metric': 'cox-nloglik'})

        train = xgb.DMatrix(X, label=y)
        val = xgb.DMatrix(X_val, label=Y_val)

        train.set_base_margin(self.predict_uni)
        val.set_base_margin(self.predict_uni_val)
        total_iterations = self.max_rounds * num_fgroups

        if self.early_stopping_rounds > 0:
            early_stopping_iterations = self.early_stopping_rounds * num_fgroups
            early_stop = xgb.callback.EarlyStopping(rounds=early_stopping_iterations,
                                                    min_delta=self.early_stopping_tolerance)
            self.booster = xgb.train(params, train, total_iterations, evals=[(val, 'eval')],
                                     verbose_eval=False,
                                     callbacks=[early_stop])
        else:
            self.booster = xgb.train(params, train, total_iterations, verbose_eval=False)

        if DEBUG_MODE:
            print(params)
            print("max iterations %d early stop iterations %d" % (
            self.max_rounds * num_fgroups, self.early_stopping_rounds * num_fgroups))
        self.time_train += time.time() - start
        start = time.time()
        self.extract_shape_functions()
        self.time_extract += time.time() - start

        self.main_fitted = self.main_fitted | len(self.interactions_chosen) == 0
        self.inter_fitted = self.main_fitted | len(self.interactions_chosen) != 0

        if len(self.interactions_chosen) == 0:
            for feature in range(self.n_features_in_):
                mean_score = np.average(self.outputs_[feature], weights=self.preprocessor_.col_bin_counts_[feature])
                self.init_params += mean_score
                self.outputs_[feature] = self.outputs_[feature] - mean_score

        start = time.time()
        self.time_detect_inter = 0
        if isinstance(interactions, int) and interactions > 0 and len(self.interactions_chosen) == 0:
            self.interactions_chosen = self.get_interactions(dataset, train, val, interactions)
        self.time_detect_inter += time.time() - start

        if DEBUG_MODE:
            print("seconds  train %f extract %f detect %f" % (self.time_train,
                                                              self.time_extract,
                                                              self.time_detect_inter))
        if self.del_booster:
            self.booster = None

        return self

    def update_limit(self, limits_dic, tree, outputs_):

        f = int(tree['split'][1:])  # tree['split_feature']
        tree_thres = int(tree['split_condition'])

        # print(int(tree['threshold']))
        if f not in limits_dic:
            limits_dic[f] = (0, outputs_[f].shape[0] - 1)

        limits_dic_left = limits_dic.copy()
        limits_dic_right = limits_dic.copy()

        limits_dic_left[f] = (limits_dic_left[f][0], tree_thres - 1)
        limits_dic_right[f] = (tree_thres, limits_dic_right[f][1])

        return limits_dic_left, limits_dic_right

    def extract_shape_functions_helper(self, tree, limits_dic={}):

        threshold_name = 'split_condition'
        leaf_name = 'leaf'

        if threshold_name in tree:
            limits_dic_left, limits_dic_right = self.update_limit(limits_dic, tree, self.outputs_)
            # print(limits_dic_left, limits_dic_right)
            left_child = tree['children'][0]
            right_child = tree['children'][1]

            self.extract_shape_functions_helper(left_child, limits_dic_left)
            self.extract_shape_functions_helper(right_child, limits_dic_right)

        else:
            keys = list(limits_dic.keys())
            if len(keys) == 1:
                f = keys[0]
                self.fit_order_curr = f # for debugging
                left_lim = max(limits_dic[f][0], 0)
                right_lim = limits_dic[f][1]
                if right_lim < self.outputs_[f].shape[0] - 1:
                    self.outputs_[f][right_lim + 1] -= tree[leaf_name]
                self.outputs_[f][left_lim] += tree[leaf_name]
            elif len(keys) == 2:
                # print(limits_dic)
                first_f, second_f = min(keys), max(keys)
                first_left_lim = max(limits_dic[first_f][0], 0)
                first_right_lim = limits_dic[first_f][1]
                sec_left_lim = max(limits_dic[second_f][0], 0)
                sec_right_lim = limits_dic[second_f][1]

                # print(tree['leaf_value'], first_left_lim,first_right_lim, sec_left_lim, sec_right_lim)
                inter_key = (first_f, second_f)
                self.fit_order_curr = inter_key # for debugging
                inter_number = self.inter_2_index[inter_key]
                if sec_right_lim < self.outputs_[second_f].shape[0] - 1:
                    #    print("minus")
                    self.outputs_inter_[inter_number][first_left_lim:first_right_lim + 1, sec_right_lim + 1] -= tree[
                        leaf_name]
                # print("plus")
                self.outputs_inter_[inter_number][first_left_lim:first_right_lim + 1, sec_left_lim] += tree[leaf_name]
            else:
                # print("wrong number of features %d" % len(keys))
                self.init_params += tree[leaf_name]
                return False
        return True

    def extract_shape_functions(self):
        """ Extract the shape functions from the trees trained by XGBoost.
        """
        import json
        self.inter_2_index = {(val[0], val[1]): i for i, val in enumerate(self.interactions_chosen)}
        if DEBUG_MODE:
            print("bin count", self.features_bin_count)
        self.outputs_ = [np.zeros(f_count, dtype="float64") for f_count in self.features_bin_count]
        self.outputs_inter_ = [
            np.zeros((self.features_bin_count_inter[inter[0]], self.features_bin_count_inter[inter[1]])) for
            inter in
            self.interactions_chosen]

        self.fit_order[self.index_stage] = []
        self.fit_order_curr = []

        model = self.booster.get_dump(dump_format='json')
        self.num_trees = len(model)
        if DEBUG_MODE:
            print("num trees %d" % self.num_trees)
        for tree_index in range(min(self.booster.best_ntree_limit, self.num_trees)):
            self.curr_tree_index = tree_index
            tree_json = json.loads(model[tree_index])
            self.extract_shape_functions_helper(tree_json, {})
            self.fit_order[self.index_stage] += [self.fit_order_curr]

        for feature in range(len(self.outputs_)):
            self.outputs_[feature] = np.cumsum(self.outputs_[feature])
        for inter_number in range(len(self.interactions_chosen)):
            self.outputs_inter_[inter_number] = np.cumsum(self.outputs_inter_[inter_number], axis=1)

    def get_interactions(self, dataset, train, valid, interactions):

        import ctypes as ct
        from interpret.glassbox.ebm.utils import EBMUtils
        X, X_inter, X_val = dataset["X"], dataset["X_inter"], dataset["X_val"]

        y, y_val = dataset["y"], dataset["Y_val"]
        iter_feature_groups = combinations(range(X_inter.shape[1]), 2)
        pair_features_categorical = np.array([x == "categorical" for x in self.preprocessor_inter.col_types_],
                                             dtype=ct.c_int64)
        pair_features_bin_count = np.array([len(x) for x in self.preprocessor_inter.col_bin_counts_], dtype=ct.c_int64)
        if 'survival' not in self.objective:
            self.predict_uni = self.predict(X)
            grads_uni = (y - self.predict_uni).astype("float64")
            if "reg" in self.model_type:
                add_init = np.mean(grads_uni)
                self.init_params += add_init
                self.predict_uni += add_init
            self.predict_uni_val = self.predict(X_val)
        else:
            grads_uni = self.booster.predict(train, pred_leaf=True).astype("float64")
            self.predict_uni = self.booster.predict(train, output_margin=True)
            self.predict_uni_val = self.booster.predict(valid, output_margin=True)

        if self.model_type == "classification":
            n_classes_ = 2
            grads = np.ascontiguousarray(y, dtype=int)
            scores = self.predict_uni.astype("float64")
        else:
            n_classes_ = -1
            grads = grads_uni
            scores = np.zeros(y.shape[0]).astype("float64").copy()

        final_indices, final_scores = EBMUtils.get_interactions(
            n_interactions=interactions,
            iter_feature_groups=iter_feature_groups,
            model_type=self.model_type,
            n_classes=n_classes_,
            features_categorical=pair_features_categorical,
            features_bin_count=pair_features_bin_count,
            X=np.ascontiguousarray(X_inter.T, dtype=int),
            y=grads,
            w=np.ascontiguousarray(np.ones(X_inter.shape[0]), dtype=float),
            scores=scores,
            min_samples_leaf=2,
        )

        return final_indices[:interactions]

    def predict(self, X):
        """
        Predict the conditional distribution of Y at the points X=x

        Parameters:
            X         : DataFrame object or List or
                        numpy array of predictors (n x p) in numeric format.
            max_iter  : get the prediction at the specified number of boosting iterations

        Output:
            A NGBoost distribution object
        """
        if isinstance(X, pd.DataFrame):
            X = X.values

        res = np.ones(len(X)) * self.init_params

        for feature_number in range(self.n_features_in_):
            res += self.outputs_[feature_number][X[:, feature_number]]

        return res


class ExplainableGBM(BaseEstimator, RegressorMixin):

    """ Explainable Boosting.
    It works exactly like the `ExplainableBoostingRegressor`/ `ExplainableBoostingClassifier` by the interpretml team,
    but here we use XGBoost as base gradient boosting model. That mean that you can choose any of the
    Learning Task Parameters of XGBoost (https://xgboost.readthedocs.io/en/stable/parameter.html).
    You can also set Monotonic restrictions in some features.
     Args:
         feature_names: List of feature names.
         max_bins: Max number of bins per feature for pre-processing stage on main effects (we use "quantile" method to bin values).
         max_interaction_bins: Max number of bins per feature for pre-processing stage on interaction terms. Only used if interactions is non-zero.
         interactions: Integer for number of automatically detected interactions.
         outer_bags: Number of outer bags.
         inner_bags: Number of inner bags. inner_bags=1 in ExplainableGBM is same as inner_bags=0 in EBM.
         learning_rate: Learning rate for boosting.
         validation_size: Validation set size for boosting.
         early_stopping_rounds: Number of rounds of no improvement to trigger early stopping.
         early_stopping_tolerance: Tolerance that dictates the smallest delta required to be considered an improvement.
         max_rounds: Number of rounds for boosting.
         max_leaves: Maximum leaf nodes used in boosting on main effects.
         max_leaves_inter: Maximum leaf nodes used in boosting interaction terms.
         objective: Specify the learning task and the corresponding learning objective.
         Examples -  'reg:squarederror' (regression with squared loss),
          'binary:logistic'(logistic regression for binary classification,
           output probability),"survival:cox"(Cox regression for right censored survival time data).
            More details here - https://xgboost.readthedocs.io/en/stable/parameter.html .
         reg_l2: Array - first value for main effect, second for interactions. L2 regularization term on weights.
         reg_l1: Array - first value for main effect, second for interactions. L1 regularization term on weights.
         feature_traverse: Array - first value for main effect, second for interactions.
         Feature group traverse methods used (main effects / interaction terms).
         * FeatureTraverse.CYCLIC - Deterministic selection by cycling through features groups one at a time.
         * FeatureTraverse.CYCLIC_RANDOM - Similar to FeatureTraverse.CYCLIC but with random feature groups shuffling for each outher bag.
         * FeatureTraverse.RANDOM- Random feature groups selection.
         * FeatureTraverse.BEST_FIT - The default behaviour in XGBoost. Gives all the features to tree.
         The tree will choose the best fitted feature.
         subsample: Subsample ratio of the training instances. Subsampling will occur once in every boosting iteration.
         monotone_constraints: Constraint of variable monotonicity.
          More details here - https://xgboost.readthedocs.io/en/stable/tutorials/monotonic.html.
         del_booster: Boolean. Delete XGboost booster. XGBoost booster can take a lot of memory and is not needed after training.
         n_jobs: Number of jobs to run in parallel.
         random_state: Random state.
     """
    def __init__(
            self,
            feature_names=None,
            max_bins: int = 256,
            max_interaction_bins: int = 32,
            interactions=10,
            outer_bags=8,
            inner_bags=1,
            learning_rate: float = 0.01,
            validation_size=0.15,
            early_stopping_rounds=50,
            early_stopping_tolerance=1e-4,
            max_rounds: Any = 5000,
            max_leaves=3,
            max_leaves_inter=8,
            objective='reg:squarederror',
            reg_l2=[0, 4],
            reg_l1=[0, 0],
            feature_traverse=[FeatureTraverse.CYCLIC_RANDOM, FeatureTraverse.CYCLIC_RANDOM],
            subsample=0.5,
            monotone_constraints=None,
            del_booster=True,
            n_jobs=-2,
            random_state=43,

    ) -> None:
        self.monotone_constraints = monotone_constraints
        self.del_booster = del_booster
        self.inner_bags = inner_bags
        self.subsample = subsample
        self.feature_traverse = feature_traverse
        self.feature_traverse = feature_traverse
        self.reg_l2 = reg_l2
        self.reg_l1 = reg_l1
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.max_leaves = max_leaves
        self.max_leaves_inter = max_leaves_inter
        self.objective = objective
        self.interactions = interactions
        self.interactions_chosen = []
        self.outer_bags = outer_bags
        self.validation_size = validation_size
        self.early_stopping_rounds = early_stopping_rounds
        self.early_stopping_tolerance = self.tol = early_stopping_tolerance
        self.max_rounds = max_rounds
        self.learning_rate = learning_rate
        self.max_bins = max_bins
        self.max_interaction_bins = max_interaction_bins
        self.outputs_ = []
        self.outputs_inter_ = []
        self.term_standard_deviations_inter_ = []
        self.term_standard_deviations_ = []
        self.init_params = []
        self.feature_names = feature_names
        self.feature_importances_ = None
        self.selector_ = None
        self.n_features_in_ = 0
        self.preprocessor_ = None
        self.preprocessor_inter = None
        self.featurs_left = []
        self.num_trees_main = []
        self.num_trees_inter = []
        self.estimators = []
        self.time_train, self.time_extract = 0, 0

    def fit(
            self, X, y: np.ndarray) \
            -> ExplainableGBM:
        """ Fits model to provided samples.

        Args:
            X: Numpy array for training samples.
            y: Numpy array as training labels.

        Returns:
            Itself.
        """
        self.outputs_ = []
        self.outputs_inter_ = []
        self.term_standard_deviations_inter_ = []
        self.term_standard_deviations_ = []
        self.init_params = []
        self.preprocessor_ = None
        self.estimators = []
        self.time_train, self.time_extract = 0, 0

        if isinstance(y, pd.Series):
            y = y.values

        if isinstance(X, pd.DataFrame):
            self.feature_names = X.columns.tolist()
            X = X.values
        elif self.feature_names is not None:
            self.feature_names = list(self.feature_names)
        else:
            self.feature_names = list(range(X.shape[1]))
        self.feature_names = [str(i) for i in self.feature_names]

        X_orig = X

        self._check_n_features(X, reset=True)

        if self.learning_rate <= 0:
            raise ValueError("learning_rate has to be positive!")

        from interpret.glassbox.ebm.ebm import EBMPreprocessor
        # converting categorical to unicode with - col_data.astype('U')
        self.preprocessor_ = EBMPreprocessor(
            feature_names=self.feature_names,
            feature_types=None,
            max_bins=self.max_bins,
            binning="quantile",
        )
        self.preprocessor_.fit(X_orig)
        X = self.preprocessor_.transform(X_orig)
        self.init_params = 0
        if "reg" in self.objective:
            self.init_params = np.mean(y)

        self.featurs_left = []

        for i in range(X.shape[1]):  # remove constant columns
            if not np.all(X[:, i] == X[0, i]):
                self.featurs_left += [i]
        X = X[:, self.featurs_left]
        self.n_features_in_ = X.shape[1]

        datasets = []
        for bag in range(self.outer_bags):
            X_, X_val_, y_, Y_val_ = train_test_split(
                X, y, test_size=self.validation_size, random_state=bag
            )
            if "reg" in self.objective:  # center the data
                y_ = y_.astype(float) - self.init_params
                Y_val_ = Y_val_.astype(float) - self.init_params
            datasets += [{"X": X_, "X_val": X_val_, "y": y_, "Y_val": Y_val_}]

        self.feature_names = [self.feature_names[x] for x in self.featurs_left]
        self.preprocessor_inter = EBMPreprocessor(
            feature_names=list(self.feature_names),
            feature_types=None,
            max_bins=self.max_interaction_bins,
            binning="quantile",
        )
        self.preprocessor_inter.fit(X_orig[:, self.featurs_left])
        X_inter = self.preprocessor_inter.transform(X_orig[:, self.featurs_left])

        for bag in range(self.outer_bags):
            datasets[bag]["X"] = datasets[bag]["X"]
            datasets[bag]["X_val"] = datasets[bag]["X_val"]
            datasets[bag]["X_inter"], datasets[bag]["X_val_inter"], _, _ = train_test_split(
                X_inter, y, test_size=self.validation_size, random_state=bag
            )
            datasets[bag]["X_inter"] = datasets[bag]["X_inter"]
            datasets[bag]["X_val_inter"] = datasets[bag]["X_val_inter"]

        self.estimators = []
        for i in range(self.outer_bags):
            self.estimators.append(ExplainableGBMCore(
                max_rounds=self.max_rounds, learning_rate=self.learning_rate, max_leaves=self.max_leaves,
                subsample=self.subsample,
                random_state=i, early_stopping_rounds=self.early_stopping_rounds,
                early_stopping_tolerance=self.early_stopping_tolerance,
                preprocessor=self.preprocessor_, preprocessor_inter=self.preprocessor_inter,
                monotone_constraints=self.monotone_constraints, objective=self.objective,
                inner_bags=self.inner_bags, reg_l2=self.reg_l2, reg_l1=self.reg_l1,
                feature_traverse=self.feature_traverse, del_booster=self.del_booster
            ))
        self.interactions_chosen = []
        self.num_trees_main = []
        self.num_trees_inter = []
        if not DEBUG_MODE:
            self.estimators = Parallel(n_jobs=self.n_jobs)(
                delayed(self.estimators[i].fit)(datasets[i], interactions=self.interactions)
                for i in range(self.outer_bags))
            # NOTE: Force gc, as Python does not free native memory easy.
            gc.collect()
        else:
            self.estimators = [self.estimators[i].fit(datasets[i], interactions=self.interactions)
                               for i in range(self.outer_bags)
                               ]

        for feature in range(self.n_features_in_):
            feature_in = []

            for estimator in self.estimators:
                if estimator.main_fitted:
                    feature_in += [estimator.outputs_[feature]]

            self.outputs_ += [np.average(feature_in, axis=0)]
            self.term_standard_deviations_ += [np.std(feature_in, axis=0)]

        self.init_params = np.mean([estimator.init_params for estimator in self.estimators])

        for estimator in self.estimators:
            self.num_trees_main += [estimator.num_trees]

        if isinstance(self.interactions, list) and len(self.interactions) > 0:
            self.interactions_chosen = self.interactions
        elif isinstance(self.interactions, int) and self.interactions > 0:
            self.interactions_chosen = self.select_pairs_from_fast_(self.estimators, self.interactions)

        if len(self.interactions_chosen) > 0:
            if DEBUG_MODE:
                print("chosen interactions", self.interactions_chosen)

            for i in range(self.outer_bags):
                self.estimators[i].interactions_chosen = self.interactions_chosen
                self.estimators[i].max_leaves = self.max_leaves_inter

            if not DEBUG_MODE:
                self.estimators = Parallel(n_jobs=self.n_jobs)(
                    delayed(self.estimators[i].fit)(datasets[i], interactions=self.interactions_chosen)
                    for i in range(self.outer_bags) if estimator.main_fitted
                )
                # NOTE: Force gc, as Python does not free native memory easy.
                gc.collect()
            else:
                self.estimators = [self.estimators[i].fit(datasets[i], interactions=self.interactions_chosen)
                                   for i in range(self.outer_bags)
                                   ]
            for estimator in self.estimators:
                self.num_trees_inter += [estimator.num_trees]

        for estimator in self.estimators:
            self.time_train += estimator.time_train / len(self.estimators)
            self.time_extract += estimator.time_extract / len(self.estimators)

            # free memory
            estimator.grads_uni = None
            estimator.grads_uni_val = None
            if self.del_booster:
                estimator.booster = None

        self.outputs_inter_ = []
        self.term_standard_deviations_inter_ = []
        if len(self.interactions_chosen) > 0:
            outputs_inter_param = []
            term_standard_deviations_param = []
            for inter, _ in enumerate(self.interactions_chosen):
                outputs_inter_estimator = []
                for estimator in self.estimators:
                    if estimator.inter_fitted:
                        outputs_inter_estimator += [estimator.outputs_inter_[inter]]

                outputs_inter_param += [np.average(outputs_inter_estimator, axis=0)]
                term_standard_deviations_param += [np.std(outputs_inter_estimator, axis=0)]

            self.outputs_inter_ = outputs_inter_param
            self.term_standard_deviations_inter_ = term_standard_deviations_param

        if len(self.interactions_chosen):
            base_names = self.feature_names.copy()
            for inter_number, inter in enumerate(self.interactions_chosen):
                self.feature_names.append(str(base_names[inter[0]]) + " x " + str(base_names[inter[1]]))

        if 'reg' in self.objective:
            self.init_params += np.mean(y - self.predict(X_orig))

        self._feature_importances_(X, X_inter)
        self.selector_ = self._selector(X_orig[:, self.featurs_left])
        return self

    @staticmethod
    def select_pairs_from_fast_(estimators, interactions):
        import heapq

        # Average rank from estimators
        inter_indices_all = []
        top_pairs_all = []
        for estimator in estimators:
            if estimator.main_fitted:
                inter_indices_all += [estimator.interactions_chosen]
        pair_ranks = {}
        for n, inter_indices_ in enumerate(inter_indices_all):
            for rank, indices in enumerate(inter_indices_):
                old_mean = pair_ranks.get(indices, 0)
                pair_ranks[indices] = old_mean + ((rank - old_mean) / (n + 1))

        final_ranks = []
        total_interactions = 0
        for indices in pair_ranks:
            heapq.heappush(final_ranks, (pair_ranks[indices], indices))
            total_interactions += 1

        top_pairs_all += [heapq.heappop(final_ranks)[1] for _ in range(min(interactions, total_interactions))]

        return top_pairs_all

    def predict(self, X, both_classes=False, bin=True):
        """ Predicts on provided samples.

        Args:
            X: Numpy array for samples.
            both_classes: If its binary classification return prediction for both classes.
            bin: Transform provided samples?.

        Returns:
            Predicted class label per sample.
        """
        if isinstance(X, pd.DataFrame):
            X = X.values

        X = X[:, self.featurs_left]
        if bin:
            X_inter = self.preprocessor_inter.transform(X)
            X = self.preprocessor_.transform(X)

        check_is_fitted(self)
        self._check_n_features(X, reset=False)

        res = np.ones(len(X)) * self.init_params

        for feature_number in range(self.n_features_in_):
            res += self.outputs_[feature_number][X[:, feature_number]]

        if self.interactions_chosen is not None and len(self.interactions_chosen) > 0:
            for inter_number, inter in enumerate(self.interactions_chosen):
                res += self.outputs_inter_[inter_number][X_inter[:, inter[0]], X_inter[:, inter[1]]]

        if 'binary' in self.objective:
            log_odds_vector = res
            if log_odds_vector.ndim == 1:
                log_odds_vector = np.c_[np.zeros(log_odds_vector.shape), log_odds_vector]
            res = softmax(log_odds_vector)
            if not both_classes:
                res = res[:, 1]
        return res

    def predict_proba(self, X):
        return self.predict(X, both_classes=True)

    def explain_local(self, X, bin=True, add_std=False):
        """ Provides local explanations for provided samples.

        Args:
            X: Numpy array for X to explain.
            bin: Transform provided samples?.
            add_std: Add standard deviation to the explanation?.

        Returns:
            An explanation object, visualizing feature-value pairs
            for each sample as horizontal bar charts.
        """
        if isinstance(X, pd.DataFrame):
            X = X.values

        X = X[:, self.featurs_left]
        if bin:
            X_inter = self.preprocessor_inter.transform(X)
            X = self.preprocessor_.transform(X)

        check_is_fitted(self)
        self._check_n_features(X, reset=False)

        res = np.ones(len(X)) * self.init_params
        explain = pd.DataFrame(res, columns=['offset'])
        explain_std = pd.DataFrame(res, columns=['offset'])

        for feature_number in range(self.n_features_in_):
            explain[self.feature_names[feature_number]] = self.outputs_[feature_number][X[:, feature_number]]
            explain_std[self.feature_names[feature_number]] = self.term_standard_deviations_[feature_number][
                X[:, feature_number]]

        if self.interactions_chosen is not None and len(self.interactions_chosen) > 0:
            for inter_number, inter in enumerate(self.interactions_chosen):
                explain[self.feature_names[self.n_features_in_ + inter_number]] = self.outputs_inter_[inter_number][
                    X_inter[:, inter[0]], X_inter[:, inter[1]]]

        if add_std:
            return explain, explain_std

        return explain

    def explain_global(self, only_inter=False):
        """Provides global explanation for model.

        Args:
            only_inter: Return only interactions.

        Returns:

            An explanation object,
            visualizing feature-value pairs as horizontal bar chart.
        """
        from interpret.glassbox.ebm.ebm import EBMExplanation
        num_inter = 0
        if len(self.interactions_chosen) > 0:
            num_inter = len(self.interactions_chosen)

        lower_bound = np.inf
        upper_bound = -np.inf
        for feature_group_index in range(self.n_features_in_):
            errors = self.term_standard_deviations_[feature_group_index]
            scores = self.outputs_[feature_group_index]

            lower_bound = min(lower_bound, np.min(scores - errors))
            upper_bound = max(upper_bound, np.max(scores + errors))

        bounds = (lower_bound, upper_bound)

        # Add per feature graph
        data_dicts = []
        feature_list = []
        density_list = []

        for feature_group_index in range(self.n_features_in_ + num_inter):

            if feature_group_index < self.n_features_in_:
                if only_inter:
                    continue

                model_graph = self.outputs_[feature_group_index]
                # NOTE: This uses stddev. for bounds, consider issue warnings.
                errors = self.term_standard_deviations_[feature_group_index]

                model_graph = model_graph[1:]
                errors = errors[1:]

                bin_labels = self.preprocessor_._get_bin_labels(feature_group_index)

                scores = list(model_graph)
                upper_bounds = list(model_graph + errors)
                lower_bounds = list(model_graph - errors)
                density_dict = {
                    "names": self.preprocessor_._get_hist_edges(feature_group_index),
                    "scores": self.preprocessor_._get_hist_counts(feature_group_index),
                }

                feature_dict = {
                    "type": "univariate",
                    "names": bin_labels,
                    "scores": scores,
                    "scores_range": bounds,
                    "upper_bounds": upper_bounds,
                    "lower_bounds": lower_bounds,
                }
                feature_list.append(feature_dict)
                density_list.append(density_dict)

                data_dict = {
                    "type": "univariate",
                    "names": bin_labels,
                    "scores": model_graph,
                    "scores_range": bounds,
                    "upper_bounds": model_graph + errors,
                    "lower_bounds": model_graph - errors,
                    "density": {
                        "names": self.preprocessor_._get_hist_edges(feature_group_index),
                        "scores": self.preprocessor_._get_hist_counts(feature_group_index),
                    },
                }
            else:
                inter_number = feature_group_index - self.n_features_in_
                feature_indexes = self.interactions_chosen[inter_number]

                model_graph = self.outputs_inter_[inter_number]
                # NOTE: This uses stddev. for bounds, consider issue warnings.
                model_graph = model_graph[1:, 1:]
                # errors = errors[1:, 1:]  # NOTE: This is commented as it's not used in this branch.

                bin_labels_left = self.preprocessor_inter._get_bin_labels(feature_indexes[0])
                bin_labels_right = self.preprocessor_inter._get_bin_labels(feature_indexes[1])

                feature_dict = {
                    "type": "interaction",
                    "left_names": bin_labels_left,
                    "right_names": bin_labels_right,
                    "scores": model_graph,
                    "scores_range": bounds,
                }
                feature_list.append(feature_dict)
                density_list.append({})

                data_dict = {
                    "type": "interaction",
                    "left_names": bin_labels_left,
                    "right_names": bin_labels_right,
                    "scores": model_graph,
                    "scores_range": bounds,
                }

            data_dicts.append(data_dict)

        if only_inter:
            overall_dict = {
                "type": "univariate",
                "names": self.feature_names[-num_inter:],
                "scores": self.feature_importances_[-num_inter:],
            }
        else:
            overall_dict = {
                "type": "univariate",
                "names": self.feature_names,
                "scores": self.feature_importances_,
            }
        internal_obj = {
            "overall": overall_dict,
            "specific": data_dicts,
            "mli": [
                {
                    "explanation_type": "ebm_global",
                    "value": {"feature_list": feature_list},
                },
                {"explanation_type": "density", "value": {"density": density_list}},
            ],
        }

        if only_inter:
            return EBMExplanation(
                "global",
                internal_obj,
                feature_names=self.feature_names[-num_inter:],
                feature_types=["interaction"] * num_inter,
                name='ExplainableLightGBM',  # + str(param_i),
                selector=self.selector_.tail(num_inter),
            )
        else:
            return EBMExplanation(
                "global",
                internal_obj,
                feature_names=self.feature_names,
                feature_types=self.preprocessor_.col_types_ + ["interaction"] * num_inter,
                name='ExplainableXGBoost',
                selector=self.selector_,
            )

    def _feature_importances_(self, X, X_inter):

        from interpret.glassbox.ebm.utils import EBMUtils
        self.feature_importances_ = []
        num_inter = 0
        if len(self.interactions_chosen) > 0:
            num_inter = len(self.interactions_chosen)

        feature_importances_param = []

        additive_terms_ = [self.outputs_[f] for f in range(len(self.outputs_))]
        feature_groups_ = [[f] for f in range(len(self.outputs_))]

        scores_gen = EBMUtils.scores_by_feature_group(
            X.T, X.T, feature_groups_, additive_terms_
        )
        for set_idx, _, scores in scores_gen:
            mean_abs_score = np.mean(np.abs(scores))
            feature_importances_param.append(mean_abs_score)

        additive_terms_ = [self.outputs_inter_[inter] for inter in range(num_inter)]
        feature_groups_ = [self.interactions_chosen[inter] for inter in range(num_inter)]

        scores_gen = EBMUtils.scores_by_feature_group(
            X_inter.T, X_inter.T, feature_groups_, additive_terms_
        )
        for set_idx, _, scores in scores_gen:
            mean_abs_score = np.mean(np.abs(scores))
            feature_importances_param.append(mean_abs_score)

        self.feature_importances_ = feature_importances_param

    def _selector(self, X_orig):

        from interpret.utils import gen_global_selector
        num_inter = 0
        if len(self.interactions_chosen) > 0:
            num_inter = len(self.interactions_chosen)

        self.feature_types = self.preprocessor_.col_types_ + ["interaction"] * num_inter
        selectors = gen_global_selector(
            X_orig, self.feature_names, self.feature_types, None
        )
        return selectors


def test_extract_shape_function():

    import pytest

    from benchmark_regression_classification import gen_synth_data_ordering_exp
    dataset = gen_synth_data_ordering_exp(num_f=5, n=1000)
    X, Y = dataset['full']['X'], dataset['full']['y']

    egbm = ExplainableGBM(interactions=0, outer_bags=1, max_bins=32, inner_bags=1, subsample=1,
                          feature_traverse=[FeatureTraverse.RANDOM, FeatureTraverse.RANDOM], del_booster=False).fit(X, Y)
    # predictions from egb
    egbm_predictions = egbm.predict(X)
    egbm_predictions = egbm_predictions + np.mean(Y - egbm_predictions)
    rmse_egbm = np.sqrt(mean_squared_error(Y, egbm_predictions))
    print(rmse_egbm)

    binned_x = egbm.preprocessor_.transform(X.values)
    dmatrix = xgb.DMatrix(binned_x)
    # predictions from xgboost
    best_ntree_limit = egbm.estimators[0].booster.best_ntree_limit
    print(best_ntree_limit, egbm.estimators[0].booster.best_iteration)
    egbm_booster_predictions = egbm.estimators[0].booster.predict(dmatrix, iteration_range=(0, egbm.estimators[0].booster.best_iteration+1))
    egbm_booster_predictions = egbm_booster_predictions + np.mean(Y - egbm_booster_predictions)
    rmse_booster = np.sqrt(mean_squared_error(Y, egbm_booster_predictions))
    print(rmse_booster)
    assert rmse_egbm == pytest.approx(rmse_booster)

    # predictions from single estimator
    egbm_xgbcore_predictions = egbm.estimators[0].predict(binned_x)
    egbm_xgbcore_predictions = egbm_xgbcore_predictions + np.mean(Y - egbm_xgbcore_predictions)
    rmse_core = np.sqrt(mean_squared_error(Y, egbm_xgbcore_predictions))
    print(rmse_core)

    assert rmse_core == pytest.approx(rmse_booster)


def test_feature_traverse():

    from benchmark_regression_classification import gen_synth_data_ordering_exp
    num_f = 5
    dataset = gen_synth_data_ordering_exp(num_f=num_f, n=1000)
    X, Y = dataset['full']['X'], dataset['full']['y']

    egbm = ExplainableGBM(interactions=0, max_rounds=2,
                          feature_traverse=[FeatureTraverse.CYCLIC_RANDOM_REVERSE, FeatureTraverse.CYCLIC_RANDOM_REVERSE],
                                    del_booster=False, outer_bags=2, inner_bags=1).fit(X, Y)
    egbm_predictions = egbm.predict(X)
    rmse_egbm_cyclic_reverse = np.sqrt(mean_squared_error(Y, egbm_predictions))

    print(egbm.estimators[0].fit_order, egbm.estimators[1].fit_order)

    # in cycle all the rounds in all thes tsimators are equal. its from difernt index in each bag so we add 1 and 2
    assert egbm.estimators[0].fit_order[0][1:num_f+1] == egbm.estimators[1].fit_order[0][2:num_f+2][::-1]

    egbm = ExplainableGBM(interactions=0, max_rounds=2,
                          feature_traverse=[FeatureTraverse.CYCLIC, FeatureTraverse.CYCLIC],
                                    del_booster=False, outer_bags=2, inner_bags=1).fit(X, Y)
    egbm_predictions = egbm.predict(X)
    rmse_egbm_cyclic = np.sqrt(mean_squared_error(Y, egbm_predictions))

    print(egbm.estimators[0].fit_order)
    print(egbm.estimators[1].fit_order)

    # in cycle all the rounds in all thes tsimators are equal
    assert egbm.estimators[0].fit_order == egbm.estimators[1].fit_order

    egbm = ExplainableGBM(interactions=0, max_rounds=2,
                          feature_traverse=[FeatureTraverse.CYCLIC_RANDOM, FeatureTraverse.CYCLIC_RANDOM],
                                    del_booster=False, outer_bags=2, inner_bags=1).fit(X, Y)
    egbm_predictions = egbm.predict(X)
    rmse_egbm_cyclic_rand = np.sqrt(mean_squared_error(Y, egbm_predictions))

    print(egbm.estimators[0].fit_order)
    print(egbm.estimators[1].fit_order)

    # in cycle all the rounds in all each estimator are equal, but rounds between estimators are different
    assert egbm.estimators[0].fit_order != egbm.estimators[1].fit_order
    assert egbm.estimators[0].fit_order[0][:num_f] == egbm.estimators[0].fit_order[0][num_f:]

    egbm = ExplainableGBM(interactions=0, max_rounds=2,
                          feature_traverse=[FeatureTraverse.RANDOM, FeatureTraverse.RANDOM],
                                    del_booster=False, outer_bags=2, inner_bags=1).fit(X, Y)
    egbm_predictions = egbm.predict(X)
    rmse_egbm_rand = np.sqrt(mean_squared_error(Y, egbm_predictions))

    print(egbm.estimators[0].fit_order)
    print(egbm.estimators[1].fit_order)

    # in random all the rounds in all  estimators are different
    assert egbm.estimators[0].fit_order != egbm.estimators[1].fit_order
    assert egbm.estimators[0].fit_order[0][:num_f] != egbm.estimators[0].fit_order[0][num_f:]

    print(rmse_egbm_rand, rmse_egbm_cyclic_rand, rmse_egbm_cyclic)

    # error should be equal
    assert rmse_egbm_rand == rmse_egbm_cyclic_rand
    assert rmse_egbm_rand == rmse_egbm_cyclic


if __name__ == '__main__':

    pass
