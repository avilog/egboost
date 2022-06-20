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


sys.path.insert(0, 'xgboost/python-package')
import xgboost as xgb
from sklearn.model_selection import train_test_split
from itertools import combinations
from sklearn.utils.extmath import softmax

DEBUG_MODE = False


class ExplainableGBMCore(BaseEstimator, RegressorMixin):

    def __init__(
            self,
            max_rounds,
            learning_rate,
            random_state,
            early_stopping_rounds,
            early_stopping_tolerance,
            validation_size,
            preprocessor,
            preprocessor_inter,
            objective,
            num_leaves,
            LightGBM_base_model,
            reg_l2,
            reg_l1,
            feature_traverse_random,
            subsample
    ) -> None:
        self.subsample = subsample
        self.feature_traverse_random = feature_traverse_random
        self.reg_l2 = reg_l2
        self.reg_l1 = reg_l1
        self.LightGBM_base_model = LightGBM_base_model
        self.num_leaves = num_leaves
        self.objective = objective
        self.model_type = "regression"
        if 'binary' in self.objective:
            self.model_type = "classification"
        self.validation_size = validation_size
        self.early_stopping_rounds = early_stopping_rounds
        self.early_stopping_tolerance = self.tol = early_stopping_tolerance
        self.max_rounds = max_rounds
        self.learning_rate = learning_rate
        self.domains_ = []
        self.outputs_ = []
        self.outputs_inter_ = []
        self.random_state = random_state
        self.init_params = 0
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

    @staticmethod
    def dmat_builder_survival(X, y):
        label_lower_bound = np.array([e[1] for e in y])
        label_upper_bound = np.array([(e[1] if e[0] else +np.inf) for e in y])
        return xgb.DMatrix(X, label_lower_bound=label_lower_bound,
                           label_upper_bound=label_upper_bound)

    def fit(
            self, dataset, interactions=None) \
            -> ExplainableGBMCore:
        """
        Fit the model.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            The training data.

        y : np.ndarray, 1-dimensional
            The target values.

        Returns
        -------
        ExplainableBoostingMetaRegressor
            Fitted regressor.
        """        """Initialize."""
        if  len(self.interactions_chosen) > 0:
            X, X_val = dataset["X_inter"], dataset["X_val_inter"]
        else:
            X, X_val = dataset["X"], dataset["X_val"]
        y, Y_val = dataset["y"], dataset["Y_val"]

        self.n_features_in_ = X.shape[1]
        if isinstance(y, pd.Series):
            y = y.values

        if 'surv' not in self.objective:
            self.init_params = np.mean(y)
        if self.model_type == "classification":
            self.init_params = 0.5

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
        index_stage = 0
        if len(self.interactions_chosen) > 0:
            index_stage = 1
        start = time.time()
        if self.LightGBM_base_model:
            sys.path.insert(0, '/home/avi/git/distributions/LightGBM/python-package')
            import lightgbm as lgb
            if self.objective == 'reg:squarederror':
                self.objective = 'regression'
            elif self.objective == "binary:logistic":
                self.objective = 'binary'#'xentropy' #'binary'
            params = {
                'max_bin': max_bin,
                'objective': self.objective,
                'num_leaves': self.num_leaves,
                'verbosity': -1,
                'colsample_bytree': 0.2 if self.feature_traverse_random[index_stage] else 1,
                'learning_rate': self.learning_rate,
                'min_child_samples': 4,
                'first_metric_only': True,
                'boost_from_average': True,
                'feature_pre_filter': False,
                'interaction_constraints': interaction_constraints_base,
                'enable_bundle': True,
                'lambda_l2': self.reg_l2[index_stage],
                'lambda_l1': self.reg_l1[index_stage],
                'seed': self.random_state,
                'bagging_fraction':  self.subsample if len(self.interactions_chosen) > 0 else 1,
            }
            eval_callback = lgb.callback.early_stopping(self.early_stopping_rounds * num_fgroups, True, verbose=False,
                                                        min_delta=self.early_stopping_tolerance)
            if len(self.interactions_chosen) > 0:
                    train = lgb.Dataset(X, y, init_score=self.predict_uni, params=params)
                    val = lgb.Dataset(X_val, Y_val, init_score=self.predict_uni_val, params=params)
            else:
                train = lgb.Dataset(X, y, params=params)
                val = lgb.Dataset(X_val, Y_val, params=params)

            total_rounds = self.max_rounds * num_fgroups
            self.booster = lgb.train(params,
                                     train,
                                     num_boost_round=total_rounds,
                                     valid_sets=[val],
                                     callbacks=[eval_callback])
        else:
            params = {'eta': self.learning_rate,
                  'objective': self.objective,
                  'max_leaves': self.num_leaves,
                  'tree_method': 'hist',
                  'max_bin': max_bin,
                  'colsample_bytree': 0.2 if self.feature_traverse_random[index_stage] else 1,
                  'colsample_bylevel': 1,
                  'colsample_bynode': 1,
                  'verbosity': 0,
                  'grow_policy': 'lossguide',
                  'min_child_weight': 4,
                  'interaction_constraints': str(interaction_constraints_base),
                  'validate_parameters': False,
                  'seed': self.random_state,
                  'seed_per_iteration': True, # from:https://www.datacamp.com/tutorial/tutorial-ridge-lasso-elastic-net
                  'alpha': self.reg_l1[index_stage], # Lasso tends to do well if there are a small number of significant parameters
                  'lambda': self.reg_l2[index_stage], # Ridge works well if there are many large parameters of about the same value
                  'subsample': self.subsample if len(self.interactions_chosen) > 0 else 1,
            }

            if self.objective == 'survival:aft':
                #params.update({'aft_loss_distribution': 'extreme',#'normal',
                #               'eval_metric': 'aft-nloglik'})
                train = self.dmat_builder_survival(X, y)
                val = self.dmat_builder_survival(X_val, Y_val)
            else:
                params['base_score'] = self.init_params
                train = xgb.DMatrix(X, label=y)
                val = xgb.DMatrix(X_val, label=Y_val)

            total_rounds = self.max_rounds * num_fgroups

            early_stopping_rounds = self.early_stopping_rounds * num_fgroups
            early_stop = xgb.callback.EarlyStopping(rounds=early_stopping_rounds,
                                                    min_delta=self.early_stopping_tolerance)
            if len(self.interactions_chosen) > 0:
                train.set_base_margin(self.predict_uni)
                val.set_base_margin(self.predict_uni_val)
            self.booster = xgb.train(params, train, total_rounds, evals=[(val, 'eval')],
                                     verbose_eval=False,
                                     callbacks=[early_stop])

        if DEBUG_MODE:
            print(params)
            print("max round %d early stop %d" % (self.max_rounds * num_fgroups, self.early_stopping_rounds * num_fgroups))
        self.time_train += time.time() - start
        start = time.time()
        self.extract_shape_functions()
        self.time_extract += time.time() - start

        self.main_fitted = self.main_fitted | len(self.interactions_chosen) == 0
        self.inter_fitted = self.main_fitted | len(self.interactions_chosen) != 0

        start = time.time()
        self.time_detect_inter = 0
        if isinstance(interactions, int) and interactions > 0 and len(self.interactions_chosen) == 0:
            if self.LightGBM_base_model:
                train = dataset["X"]
                val = dataset["X_val"]
            self.interactions_chosen = self.get_interactions(dataset, train, val, interactions)
        self.time_detect_inter += time.time() - start

        if self.model_type == "regression" and len(self.interactions_chosen) == 0:
            for feature in range(self.n_features_in_):
                mean_score = np.average(self.outputs_[feature], weights=self.preprocessor_.col_bin_counts_[feature])
                self.init_params += mean_score
                self.outputs_[feature] = self.outputs_[feature] - mean_score

        if DEBUG_MODE:
            print("seconds  train %f extract %f detect %f" % (self.time_train,
                                                                       self.time_extract,
                                                                       self.time_detect_inter))

        return self

    def update_limit(self, limits_dic, tree, outputs_):

        if self.LightGBM_base_model:
            f = tree['split_feature']
            tree_thres = int(tree['threshold'])
        else:
            f = int(tree['split'][1:])  # tree['split_feature']
            tree_thres = int(tree['split_condition'])
        # print(int(tree['threshold']))
        if f not in limits_dic:
            limits_dic[f] = (0, outputs_[f].shape[0] - 1)

        limits_dic_left = limits_dic.copy()
        limits_dic_right = limits_dic.copy()

        if self.LightGBM_base_model:
            limits_dic_left[f] = (limits_dic_left[f][0], tree_thres)
            limits_dic_right[f] = (tree_thres + 1, limits_dic_right[f][1])
        else:
            limits_dic_left[f] = (limits_dic_left[f][0], tree_thres - 1)
            limits_dic_right[f] = (tree_thres, limits_dic_right[f][1])

        return limits_dic_left, limits_dic_right

    def extract_shape_functions_helper(self, tree, limits_dic={}):

        if self.LightGBM_base_model:
            threshold_name = 'threshold'
            leaf_name = 'leaf_value'
        else:
            threshold_name = 'split_condition'
            leaf_name = 'leaf'

        if threshold_name in tree:
            limits_dic_left, limits_dic_right = self.update_limit(limits_dic, tree, self.outputs_)
            # print(limits_dic_left, limits_dic_right)
            if self.LightGBM_base_model:
                left_child = tree['left_child']
                right_child = tree['right_child']
            else:
                left_child = tree['children'][0]
                right_child = tree['children'][1]

            self.extract_shape_functions_helper(left_child, limits_dic_left)
            self.extract_shape_functions_helper(right_child, limits_dic_right)

        else:
            keys = list(limits_dic.keys())
            if len(keys) == 1:
                f = keys[0]
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

        import json
        self.inter_2_index = {(val[0], val[1]): i for i, val in enumerate(self.interactions_chosen)}
        if DEBUG_MODE:
            print("bin count", self.features_bin_count)
        self.outputs_ = [np.zeros(f_count) for f_count in self.features_bin_count]+[np.zeros(f_count) for f_count in self.features_bin_count_inter]
        self.outputs_inter_ = [np.zeros((self.features_bin_count_inter[inter[0]], self.features_bin_count_inter[inter[1]])) for
                               inter in
                               self.interactions_chosen]

        if self.LightGBM_base_model:
            model = self.booster.dump_model()
            self.num_trees = self.booster.num_trees()
            if DEBUG_MODE:
                print("num trees %d best iteration %d len model info %d" % (self.num_trees, self.booster.best_iteration, len(model['tree_info'])))
            for tree_index in range(min(self.booster.best_iteration, self.num_trees)):#self.booster.best_iteration
                tree_structure = model['tree_info'][tree_index]['tree_structure']
                self.extract_shape_functions_helper(tree_structure, {})
        else:
            model = self.booster.get_dump(dump_format='json')
            self.num_trees = len(model)
            if DEBUG_MODE:
                print("num trees %d" % self.num_trees)
            for tree_index in range(self.booster.best_iteration):#self.booster.best_iteration+1
                tree_json = json.loads(model[tree_index])
                self.extract_shape_functions_helper(tree_json, {})

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
        if self.objective != 'survival:aft':
            self.predict_uni = self.predict(X)
            grads_uni = (y-self.predict_uni).astype("float64")
            add_init = 0
            if self.model_type != "classification":
                add_init = np.mean(grads_uni)
                self.init_params += add_init
            grads_uni -= add_init
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
    """
    A meta regressor that outputs a transparent, explainable model given blackbox models.

    It works exactly like the `ExplainableBoostingRegressor` by the interpretml team, but here you can choose any base regressor instead of
    being restricted to trees. For example, you can use scikit-learn's `IsotonicRegression` to create a model that is
    monotonically increasing or decreasing in some of the features, while still being explainable and well-performing.

    See the notes below to find a nice explanation of how the algorithm works at a high level.

    Parameters
    ----------
    base_regressor : Any, default=DecisionTreeRegressor(max_leaves=3)
        A single scikit-learn compatible regressor

    max_rounds : int, default=2000
        Conduct the boosting for these many rounds.

    learning_rate : float, default=0.01
        The learning rate. Should be quite small.

    max_bins : int, default=256
        The more grid points, the

            - more detailed the explanations get and
            - the better the model performs, but
            - the slower the algorithm gets.

    Notes
    -----
    Check out the original author's Github at https://github.com/interpretml/interpret and https://www.youtube.com/watch?v=MREiHgHgl0k
    for a great introduction into the operations of the algorithm.
    """

    def __init__(
            self,
            max_rounds: Any = 5000,
            learning_rate: float = 0.01,
            max_bins: int = 256,
            max_interaction_bins: int = 32,
            feature_names=None,
            early_stopping_rounds=50,
            early_stopping_tolerance=1e-4,
            validation_size=0.15,
            outer_bags=8,
            interactions=10,
            objective='reg:squarederror',
            num_leaves=5,
            num_leaves_inter=9,
            n_jobs=-2,
            random_state=43,
            LightGBM_base_model=False,
            reg_l2=[0, 4],
            reg_l1=[0, 0],
            feature_traverse_random=[True, True],
            subsample=0.6
    ) -> None:
        self.subsample = subsample
        self.feature_traverse_random = feature_traverse_random
        self.reg_l2 = reg_l2
        self.reg_l1 = reg_l1
        self.LightGBM_base_model = LightGBM_base_model
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.num_leaves = num_leaves
        self.num_leaves_inter = num_leaves_inter
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
        self.estimators = []
        self.time_train, self.time_extract = 0, 0

    def fit(
            self, X, y: np.ndarray) \
            -> ExplainableGBM:
        """
        Fit the model.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            The training data.

        y : np.ndarray, 1-dimensional
            The target values.

        Returns
        -------
        ExplainableBoostingMetaRegressor
            Fitted regressor.
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
            feature_names=self.feature_names,
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
                max_rounds=self.max_rounds, learning_rate=self.learning_rate, num_leaves=self.num_leaves, subsample=self.subsample,
                random_state=i, early_stopping_rounds=self.early_stopping_rounds, early_stopping_tolerance=self.early_stopping_tolerance,
                validation_size=self.validation_size, preprocessor=self.preprocessor_, preprocessor_inter=self.preprocessor_inter,
                objective=self.objective, LightGBM_base_model=self.LightGBM_base_model,
                reg_l2=self.reg_l2, reg_l1=self.reg_l1, feature_traverse_random=self.feature_traverse_random
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
                self.estimators[i].num_leaves = self.num_leaves_inter
                #self.estimators[i].preprocessor = self.preprocessor_inter

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

        if 'binary' not in self.objective:
            if 'survival' not in self.objective:
                self.init_params += np.mean(y-self.predict(X_orig))
            else:
                self.init_params += np.mean(y["Time"] - self.predict(X_orig))
        else:
            self.init_params = 0
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

    def predict(self, X, raw=False, bin = True):
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
            if not raw:
                res = res[:, 1]#.round().astype(int)
        return res

    def predict_proba(self, X):
        return self.predict(X, raw=True)

    def explain_local(self, X, bin = True):
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

        X = X[:, self.featurs_left]
        if bin:
            X_inter = self.preprocessor_inter.transform(X)
            X = self.preprocessor_.transform(X)

        check_is_fitted(self)
        self._check_n_features(X, reset=False)

        res = np.ones(len(X)) * self.init_params
        explain = pd.DataFrame(res, columns=['offset'])
        for feature_number in range(self.n_features_in_):
            explain[self.feature_names[feature_number]] = self.outputs_[feature_number][X[:, feature_number]]

        if self.interactions_chosen is not None and len(self.interactions_chosen) > 0:
            for inter_number, inter in enumerate(self.interactions_chosen):
                explain[self.feature_names[self.n_features_in_+inter_number]] = self.outputs_inter_[inter_number][X_inter[:, inter[0]], X_inter[:, inter[1]]]

        return explain

    def explain_global(self, only_inter=False):
        """Provides global explanation for model.
        Args:
            name: User-defined explanation name.
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

                bin_labels_left = self.preprocessor_._get_bin_labels(feature_indexes[0])
                bin_labels_right = self.preprocessor_._get_bin_labels(feature_indexes[1])

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
                name='ExplainableLightGBM' if self.LightGBM_base_model else 'ExplainableXGBoost',
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

        self.feature_types = self.preprocessor_.col_types_ + ["interaction"]*num_inter
        selectors = gen_global_selector(
            X_orig, self.feature_names, self.feature_types, None
        )
        return selectors


if __name__ == '__main__':

    from utils_data import *
    import numpy as np
    pd.set_option('display.max_rows', 500)
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 1000)

    """
    X, Y, x_to_loc, x_to_scale = gen_synth_data(noise_c=0, add_inter=True, scale_constant=False, n=500)
    X_train, X_val, y_train, y_val = train_test_split(
        X, Y, test_size=0.2, random_state=42)

    elightgbm_ = ExplainableGBM(interactions=10, outer_bags=1, num_leaves=3, max_bins=256, LightGBM_base_model=True). \
        fit(X_train, y_train)
    rmse_ebm = np.sqrt(mean_squared_error(y_val, elightgbm_.predict(X_val)))
    print(rmse_ebm)

    from interpret.glassbox import ExplainableBoostingRegressor

    ebm = ExplainableBoostingRegressor(interactions=10, outer_bags=1, max_bins=256). \
        fit(X_train, y_train)
    rmse_ebm = np.sqrt(mean_squared_error(y_val, ebm.predict(X_val)))
    print(rmse_ebm)
    """

    """
    from sklearn.datasets import make_classification  # Easy decision boundary
    from sklearn.metrics import roc_auc_score

    X, y = make_classification(n_samples=1000, n_features=8, n_informative=7, n_redundant=0, n_repeated=0, n_classes=2,
                               n_clusters_per_class=4, class_sep=1, flip_y=0, random_state=18)
    objective = 'binary:logistic'
    ebm_model = ExplainableGBM(objective=objective, base_model="LightGBM", interactions=10)

    ebm_model.fit(X, y)
    y_predicted = ebm_model.predict_proba(X)[:, 1]
    print('AUC = %.2f' % roc_auc_score(y, y_predicted))

    from interpret.glassbox import  ExplainableBoostingClassifier

    ebm_model = ExplainableBoostingClassifier(n_jobs=-1, interactions=10)
    ebm_model.fit(X, y)
    y_predicted = ebm_model.predict_proba(X)[:, 1]
    print('AUC = %.2f' % roc_auc_score(y, y_predicted))

    ebm_model = ExplainableBoostingClassifier(n_jobs=-1, interactions=0)
    ebm_model.fit(X, y)
    y_predicted = ebm_model.predict_proba(X)[:, 1]
    print('AUC = %.2f' % roc_auc_score(y, y_predicted))
    """

    X, Y, x_to_loc, _ = gen_synth_data(noise_c=0, add_inter=True, n=5000)
    X_train, X_val, y_train, y_val = train_test_split(
                    X, Y, test_size=0.2, random_state=42)

    egbm_model_bf = ExplainableGBM(interactions=0, LightGBM_base_model=False,
                                             feature_traverse_random=False).fit(X_train, y_train)

    """
    def compute_all_imp(ebm):
        names = ebm.feature_names
        feature_importances_ = ebm.feature_importances_

        sum_all_imp = 0
        sum_noise_imp = 0
        sum_uni_imp = 0
        sum_uni_noise_imp = 0
        for i in range(len(feature_importances_)):
            sum_all_imp += feature_importances_[i]
            if not " x " in names[i]:
                sum_uni_imp += feature_importances_[i]
            real_names = ["x" + str(i + 1) for i in range(6)] + ["x2 x x6", "x3 x x4", "x3 x x5"]
            if names[i] not in real_names:
                sum_noise_imp += feature_importances_[i]
                if " x " not in names[i]:
                    sum_uni_noise_imp += feature_importances_[i]

        return sum_all_imp, sum_noise_imp, sum_uni_imp, sum_uni_noise_imp

    def get_error_bias_variance(test_y_pred, y_test):
        variance = np.mean(np.var(test_y_pred))
        avg_test_y_pred = np.mean(test_y_pred)
        bias = np.mean((avg_test_y_pred - y_test) ** 2)
        error_mse = mean_squared_error(test_y_pred, y_test)

        return variance, bias, error_mse


    l1_imp_error = []
    range_vals = [0, 1, 2, 5, 10, 20]
    for l2_inter in range_vals:
        for l1_inter in range_vals:
            if l2_inter/5 < 1:
                continue
            for l1 in range_vals:
                for l2 in range_vals:
                    l1_ = l1 / 5
                    l2_ = l2 / 5
                    l1_inter_ = l1_inter / 5
                    l2_inter_ = l2_inter / 5
                    print({"l1": l1_, "l2": l2_, "l1_inter":l1_inter_, "l2_inter":l2_inter_})

                    egbm_model = ExplainableGBM(reg_l1=[l1_, l1_inter_], reg_l2=[l2_, l2_inter_], interactions=0, LightGBM_base_model=False).fit(
                        X_train, y_train)
                    variance_uni, bias_uni, error_mse_uni = get_error_bias_variance(egbm_model.predict(X_val), y_val)

                    egbm_model = ExplainableGBM(reg_l1=[l1_, l1_inter_], reg_l2=[l2_, l2_inter_], interactions=10, LightGBM_base_model=False).fit(
                        X_train, y_train)
                    variance, bias, error_mse = get_error_bias_variance(egbm_model.predict(X_val), y_val)

                    sum_all_imp, sum_noise_imp, sum_uni_imp, sum_uni_noise_imp = compute_all_imp(egbm_model)
                    print(sum_noise_imp / sum_all_imp)
                    l1_imp_error += [
                        {"l1": l1_, "l2": l2_, "l1_inter":l1_inter_, "l2_inter":l2_inter_, "sum_all_imp": sum_all_imp,
                         "sum_noise_imp": sum_noise_imp, "sum_uni_imp": sum_uni_imp,
                         "sum_uni_noise_imp": sum_uni_noise_imp, "error_mse": error_mse,"error_mse_uni": error_mse_uni, "variance":variance,
                         "variance_uni": variance_uni, "bias": bias, "bias_uni": bias_uni,
                         "trees_main":np.mean(egbm_model.num_trees_main),
                         "num_trees_inter":np.mean(egbm_model.num_trees_inter)}]

                    l1_imp_error_df = pd.DataFrame(l1_imp_error)
                    l1_imp_error_df["%noise_imp"] = (l1_imp_error_df["sum_noise_imp"] / l1_imp_error_df["sum_all_imp"]) * 100
                    l1_imp_error_df["%noise_inter_imp"] = ((l1_imp_error_df["sum_noise_imp"] - l1_imp_error_df["sum_uni_noise_imp"]) / (
                                l1_imp_error_df["sum_all_imp"] - l1_imp_error_df["sum_uni_imp"])) * 100
                    l1_imp_error_df["%noise_uni_imp"] = (l1_imp_error_df["sum_uni_noise_imp"] / l1_imp_error_df["sum_uni_imp"]) * 100

                    l1_imp_error_df.to_csv("l1_imp_error4.csv")
    """
