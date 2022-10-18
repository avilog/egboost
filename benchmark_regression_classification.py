import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import ShuffleSplit, StratifiedShuffleSplit
from sklearn.model_selection import ParameterGrid
from interpret.glassbox import ExplainableBoostingRegressor, ExplainableBoostingClassifier
import sys
import time
from pathlib import Path

sys.path.insert(0, 'xgboost_compiled/python-package')
import xgboost as xgb
import egboost
import pickle
from pygam import LogisticGAM, LinearGAM, f, s
from sklearn.base import BaseEstimator
from sklearn.model_selection import train_test_split
from datetime import datetime

RANDOM_STATE = 42
N_SPLITS = 5
N_ROUNDS = 30000
EARLY_STOP = 50
N_SUBSAMPLES = 5
SUBSAMPLE_RATIO = 0.5


class GAMModel(BaseEstimator):
    def __init__(
            self,
            problem,
            val_size=0.176,
            mylam=None
    ):
        self.classes_ = 1
        self.model = None
        self.problem = problem
        if self.problem == "classification":
            self.classes_ = 2
        self.max_iter = 500
        self.n_splines = 30
        self.val_size = val_size
        self.mylam = mylam

    def fit(
            self,
            X,
            y,
    ):
        formulas = []
        dtypes = []
        for idx in range(X.shape[1]):
            uniq = np.unique(X[:, idx])
            num_unique_x = len(uniq)
            if num_unique_x == 2 and np.max(uniq) == 1 and np.min(uniq) == 0:  # If binary
                # dtypes.append('categorical')
                formulas.append(f(idx))
            else:
                formulas.append(s(idx))
            dtypes.append('numerical')

        the_formula = formulas[0]
        for term in formulas[1:]:
            the_formula += term

        if self.mylam is None:
            self.mylam = [1] * len(the_formula)

        if self.problem == "classification":
            # increasing regularization to 1 is needed for converging on support dataset
            self.model = LogisticGAM(the_formula, dtype=dtypes, max_iter=self.max_iter, n_splines=self.n_splines,
                                      lam=self.mylam)
        else:
            self.model = LinearGAM(the_formula, dtype=dtypes, max_iter=self.max_iter, n_splines=self.n_splines,
                                   lam=self.mylam)

        import pygam
        # do a grid search over here
        if self.mylam is None:
            mylam = np.logspace(-1, 1, 15)
            try:
                print('search range from %f to %f' % (mylam[0], mylam[-1]))
                self.model.gridsearch(X, y, lam=mylam)
            except (np.linalg.LinAlgError, pygam.utils.OptimizationError) as e:
                print('Get the following error:', str(e), '\nRetry the grid search')
                if hasattr(self.model, 'coef_'):
                    del self.model.coef_

                self._fit(X, y, mylam=mylam[1:])
        else:
            self.model.fit(X, y)

        return self

    def predict(self, X):
        return self.model.predict(X)

    def decision_function(self, X):
        return self.model.predict_proba(X)

    def predict_proba(self, X):
        result = self.model.predict_proba(X)
        return np.vstack((1. - result, result)).transpose()


class XGMModel(BaseEstimator):
    def __init__(
            self, problem, val_size=0.176,  # 85% * 0.176 = 15%
            max_depth=3, subsample=0.5, num_parallel_tree=10, reg_l2=1,
            monotone_constraints=None, n_rounds=N_ROUNDS, early_stop=EARLY_STOP
    ):
        self.model = None
        self.problem = problem
        self.classes_ = 1
        if self.problem == "classification":
            self.classes_ = 2
        self.val_size = val_size
        self.max_depth = max_depth
        self.subsample = subsample
        self.num_parallel_tree = num_parallel_tree
        self.reg_l2 = reg_l2
        self.monotone_constraints = monotone_constraints
        self.n_rounds = n_rounds
        self.early_stop = early_stop

    def fit(
            self,
            X,
            y,
    ):
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=self.val_size
        )
        params = {'eta': 0.1,
                  'tree_method': 'hist',
                  'verbosity': 0,
                  'grow_policy': 'lossguide',
                  'max_depth': self.max_depth,
                  'random_state': RANDOM_STATE,
                  'lambda': self.reg_l2
                  }
        if self.subsample < 1:
            params['subsample'] = self.subsample
            params['num_parallel_tree'] = self.num_parallel_tree

        if self.monotone_constraints is not None:
            params['monotone_constraints'] = '(' + ','.join([str(m) for m in self.monotone_constraints]) + ')'

        if self.problem == "classification":
            params['objective'] = 'binary:logistic'
        elif self.problem == "survival":
            params['objective'] = "survival:cox"

        train = xgb.DMatrix(X_train, label=y_train)
        val = xgb.DMatrix(X_val, label=y_val)

        early_stopping_tolerance = 1e-4
        early_stop = xgb.callback.EarlyStopping(rounds=self.early_stop,
                                                min_delta=early_stopping_tolerance)

        self.model = xgb.train(params, train, self.n_rounds, evals=[(val, 'eval')],
                               verbose_eval=False,
                               callbacks=[early_stop])
        return self

    def predict(self, X):
        test = xgb.DMatrix(X)
        return self.model.predict(test)

    def decision_function(self, X):
        test = xgb.DMatrix(X)
        return self.model.predict(test)

    def predict_proba(self, X):
        test = xgb.DMatrix(X)
        return self.model.predict(test)


def get_abs_dir(rel_path):
    return Path(__file__).resolve().parent / Path(rel_path)


def add_dataset_stats(dataset):
    dataset['num_rows'] = dataset['full']['X'].shape[0]
    dataset['num_features'] = dataset['full']['X'].shape[1]
    from interpret.glassbox.ebm.ebm import EBMPreprocessor

    preprocessor = EBMPreprocessor(
        feature_types=None,
        max_bins=256,
        binning="quantile",
    )
    preprocessor.fit(dataset['full']['X'].values)
    dataset['mean_bins_256'] = format_n(np.mean(np.array([len(x) for x in preprocessor.col_bin_counts_])))

    preprocessor_inter = EBMPreprocessor(
        feature_types=None,
        max_bins=32,
        binning="quantile",
    )
    preprocessor_inter.fit(dataset['full']['X'].values)
    dataset['mean_bins_32'] = format_n(np.mean(np.array([len(x) for x in preprocessor_inter.col_bin_counts_])))
    dataset['Positive Rate'] = 0
    if dataset['problem'] == 'classification':
        dataset['Positive Rate'] = dataset['full']['y'][dataset['full']['y'] == 1].shape[0] / \
                                   dataset['full']['y'].shape[0]

    return dataset


def load_compas_data():
    # COMPAS: https://www.kaggle.com/danofer/compass
    df = pd.read_csv(get_abs_dir('data/uci/propublica_data_for_fairml.csv'))
    TARGET_COL = "Two_yr_Recidivism"
    X = df.drop([TARGET_COL], axis=1)
    y = df[TARGET_COL]
    dataset = {
        'dataset_name': 'compass',
        'problem': 'classification',
        'full': {
            'X': X,
            'y': y,
        },
    }
    return add_dataset_stats(dataset)


def load_adult_data():
    # df = pd.read_csv(
    #    "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data",
    #    header=None)
    df = pd.read_csv(
        get_abs_dir("data/uci/adult.data"),
        header=None)
    df.columns = [
        "Age", "WorkClass", "fnlwgt", "Education", "EducationNum",
        "MaritalStatus", "Occupation", "Relationship", "Race", "Gender",
        "CapitalGain", "CapitalLoss", "HoursPerWeek", "NativeCountry", "Income"
    ]
    train_cols = df.columns[0:-1]
    label = df.columns[-1]
    X_df = df[train_cols]
    df[label] = df[label].str.strip()
    y_df = df[label].map({'>50K': 1, '<=50K': 0})
    dataset = {
        'dataset_name': 'adult',
        'problem': 'classification',
        'full': {
            'X': X_df,
            'y': y_df,
        },
    }

    return add_dataset_stats(dataset)


def load_telco_churn_data():
    # https://www.kaggle.com/blastchar/telco-customer-churn/downloads/WA_Fn-UseC_-Telco-Customer-Churn.csv/1
    df = pd.read_csv(get_abs_dir('data/surv/telco_churn.csv'))
    # small number of the values are not recognized as numbers, so binning it will give us all the unique values (6532)
    # It increase the running of learning interactions in EBM by alot (10 sec before fixing, 1 sec after)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    #print("before dropping na %d" % df.shape[0])
    df = df.dropna()
    #print("after dropping na %d" % df.shape[0])

    train_cols = df.columns[1:-1]  # First column is an ID
    label = df.columns[-1]
    X_df = df[train_cols]
    y_df = df[label].map({'Yes': 1, 'No': 0})  # 'Yes, No'
    dataset = {
        'dataset_name': 'churn',
        'problem': 'classification',
        'full': {
            'X': X_df,
            'y': y_df,
        },
    }

    return add_dataset_stats(dataset)


def load_fico_score_data():
    """Loads the FICO Score dataset.
    The FICO score is a widely used proprietary credit score todetermine credit
    worthiness for loans in the United States. The FICO dataset is comprised of
    real-world anonymized credit applications made by customers and their assigned
    FICO Score, based on their credit report information. For more info, refer to
    https://community.fico.com/s/explainable-machine-learning-challenge.
    Returns:
    A dict containing the `problem` type (i.e. regression) and the
    input features `X` as a pandas.Dataframe and the FICO scores `y` as
    np.ndarray.
    """

    # Create column names
    attributes = open(get_abs_dir('data/fico/fico_score.attr'), 'r').readlines()
    column_names = [x.split(':')[0] for x in attributes]

    df = pd.read_csv(
        'data/fico/fico_score.data',
        header=None,
        names=column_names,
        delim_whitespace=True)
    train_cols = column_names[:-1]
    label = column_names[-1]
    X_df = df[train_cols]
    y_df = df[label]

    dataset = {
        'dataset_name': 'fico',
        'problem': 'regression',
        'full': {
            'X': X_df,
            'y': y_df,
        },
    }
    return add_dataset_stats(dataset)


def load_breast_data():
    from sklearn.datasets import load_breast_cancer
    breast = load_breast_cancer()
    feature_names = list(breast.feature_names)
    X, y = pd.DataFrame(breast.data, columns=feature_names), pd.Series(breast.target)
    dataset = {
        'problem': 'classification',
        'full': {
            'X': X,
            'y': y,
        },
        'dataset_name': 'breast',
    }
    return dataset


def load_support2_data():
    support = pd.read_csv(get_abs_dir('data/support2/support2.csv'))
    X_df = support.drop('hospdead', axis=1)
    y_df = support['hospdead']

    remove_features = ['death', 'slos', 'd.time', 'dzgroup', 'charges', 'totcst',
                       'totmcst', 'aps', 'sps', 'surv2m', 'surv6m', 'prg2m', 'prg6m',
                       'dnr', 'dnrday', 'avtisst', 'sfdm2']

    X_df = X_df.drop(remove_features, axis=1)
    one_hot_encode_cols = ['sex', 'dzclass', 'race', 'ca', 'income']

    rest_colmns = [c for c in X_df.columns if c not in one_hot_encode_cols]
    # Impute the missing values for 0.
    X_df[rest_colmns] = X_df[rest_colmns].fillna(0.)

    X_df['income'][X_df['income'].isna()] = 'NaN'
    X_df['income'][X_df['income'] == 'under $11k'] = ' <$11k'
    X_df['race'][X_df['race'].isna()] = 'NaN'

    dataset = {
        'dataset_name': 'support2',
        'problem': 'classification',
        'full': {
            'X': X_df,
            'y': y_df,
        },
    }

    return add_dataset_stats(dataset)


def load_bike_sharing_data():
    data = pd.read_csv(get_abs_dir("data/uci/Bike-Sharing-Dataset/hour.csv")).\
        drop(["instant", "dteday", "casual", "registered"], axis=1)
    X, y = data.iloc[:, :-1], data.iloc[:, -1]
    dataset = {
        'dataset_name': 'bike',
        'problem': 'regression',
        'full': {
            'X': X,
            'y': y,
        },
    }
    return add_dataset_stats(dataset)


def load_california_housing_data():
    """Loads the California Housing dataset.
    California  Housing  dataset is a canonical machine learning dataset derived
    from the 1990 U.S. census to understand the influence of community
    characteristics on housing prices. The task is regression to predict the
    median price of houses (in million dollars) in each district in California.
    For more info, refer to
    https://scikit-learn.org/stable/datasets/index.html#california-housing-dataset.
    """
    feature_names = [
        'Latitude', 'Longitude', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup', 'MedInc',
        'medianHouseValue'
    ]

    df = pd.read_csv(get_abs_dir("data/cal_housing.data"))
    df.columns = feature_names

    target_col = df.columns[-1]
    df[target_col] /= 100000.0  # Like in NAM paper
    X, y = df.drop([target_col], axis=1), df[target_col]

    dataset = {
        'dataset_name': 'california housing',
        'problem': 'regression',
        'full': {
            'X': X,
            'y': y,
        },
    }
    return add_dataset_stats(dataset)


def load_california_housing_data_monotone():
    dataset = load_california_housing_data()
    dataset['problem'] = 'regression_monotone'
    dataset['dataset_name'] = 'california housing monotone'

    dataset['monotone_constraints'] = [0]*dataset['full']['X'].shape[1]# 0 = no constraints
    dataset['monotone_constraints'][7] = 1  # 'MedInc'

    return dataset


def load_wine_data():
    df = pd.read_csv(get_abs_dir("data/uci/winequality-white.csv"), delimiter=';')
    target_col = 'quality'
    X, y = df.drop([target_col], axis=1), df[target_col]

    dataset = {
        'dataset_name': 'wine',
        'problem': 'regression',
        'full': {
            'X': X,
            'y': y,
        },
    }
    return add_dataset_stats(dataset)


def load_crimes_data():
    # Data from:http://archive.ics.uci.edu/ml/datasets/Communities+and+Crime.
    #https://github.com/vbordalo/Communities-Crime/blob/master/Crime_v1.ipynb
    attrib = pd.read_csv(get_abs_dir('data/uci/communities_attributes.csv'), delim_whitespace=True)
    data = pd.read_csv(get_abs_dir('data/uci/communities.data'), names=attrib['attributes'])
    data = data.drop(columns=['state', 'county',
                              'community', 'communityname',
                              'fold'], axis=1)
    data = data.replace('?', np.nan).astype(float)

    data['OtherPerCap'] = data['OtherPerCap'].fillna(value=data['OtherPerCap'].astype(float).mean())
    data = data.dropna(axis=1)
    target = "ViolentCrimesPerPop"
    X = data.drop([target], axis=1)
    y = data[target]#100

    dataset = {
        'dataset_name': 'Crimes',
        'problem': 'regression',
        'full': {
            'X': X,
            'y': y,
        },
    }
    return add_dataset_stats(dataset)


def load_nhanesi_data():
    # Data from: https://wwwn.cdc.gov/nchs/nhanes/nhanes1/ with mortality data from the NHANES I Epidemiologic Followup Study.
    # packaged data from: https://github.com/slundberg/shap/tree/master/data
    X = pd.read_csv(get_abs_dir("data/nahnesi/NHANESI_X.csv"), index_col=0)
    X["sex_isFemale"] = X["sex_isFemale"].replace({True: "Female", False: "Male"})
    X = X.rename({"sex_isFemale": "sex"}, axis=1)
    y = pd.read_csv(get_abs_dir("data/nahnesi/NHANESI_y.csv"), index_col=0)["y"]
    dataset = {
        'dataset_name': 'NHANES I',
        'problem': 'survival',
        'full': {
            'X': X,
            'y': y,
        },
    }
    return add_dataset_stats(dataset)


def gen_synth_data_ordering_exp(num_f=100, n=1000):

    import scipy
    noise_c = 0
    x_to_loc = {
        "x1": lambda x:  np.sin(np.multiply(x, 6))}
    X_df = pd.DataFrame({"x" + str(i + 1): scipy.stats.uniform.rvs(size=n, random_state=RANDOM_STATE) for i in range(num_f)})
    for i in range(num_f):
        X_df["x" + str(i + 1)] = X_df["x1"]

    loc = (sum([x_to_loc[name](X_df[name]) for name in x_to_loc.keys()]))
    noise = np.random.normal(scale=noise_c, size=n)
    y = loc + noise
    dataset = {
        'dataset_name': 'synthetic_ordering'+str(num_f),
        'problem': 'regression',
        'full': {
            'X': X_df,
            'x_to_loc': x_to_loc,
            'y': y,
        },
    }
    return add_dataset_stats(dataset)


def gen_synth_data_correlation_exp():
    import scipy
    noise_c = 0
    n = 1000
    x_to_loc = {
        "x1": lambda x: 0.5 * np.sin(np.multiply(x, 6)) + 0.5 * ((np.array(x) > 0.5) & (np.array(x) < 0.52)).astype(
            float)}
    X_df = pd.DataFrame({"x" + str(i + 1): scipy.stats.uniform.rvs(size=n, random_state=i) for i in range(5)})
    X_df["x2"] = X_df["x1"]
    X_df["x4"] = X_df["x5"]

    x_inter_loc = X_df["x3"] * X_df["x4"]
    loc = (sum([x_to_loc[name](X_df[name]) for name in x_to_loc.keys()]) + x_inter_loc)
    noise = np.random.normal(scale=noise_c, size=n)
    y = loc + noise
    dataset = {
        'dataset_name': 'synthetic_corr',
        'problem': 'regression',
        'full': {
            'X': X_df,
            'x_to_loc': x_to_loc,
            'y': y,
        },
    }
    return add_dataset_stats(dataset)


def format_n(x):
    return "{0:.3f}".format(x)


def save_load_model(dataset, name, split_idx, model=None, output_dir="models"):
    import os
    model_path = get_abs_dir(output_dir / Path('%s_%s_%d.pkl' % (dataset, name, split_idx)))
    if model is not None:
        pickle.dump(model, open(model_path, 'wb'))
    else:
        try:
            with open(model_path, 'rb') as fp:
                return pickle.load(fp)
        except Exception as e:  # work on python 3.x
            print(e)
            return None


def get_shuffle_split(problem, n_splits, random_state, test_size=0.15):
    # Evaluate model
    if problem == 'classification':
        ss = StratifiedShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=random_state)
    else:
        ss = ShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=random_state)

    return ss


def get_model(params, problem, X, ct=None):
    val_size = 0.176  # 85% * 0.176 = 15%
    if ct is None:
        is_cat = np.array([dt.kind == 'O' for dt in X.dtypes])
        cat_cols = X.columns.values[is_cat]
        num_cols = X.columns.values[~is_cat]

        # We need to do it here, because we have problems converging on adult dataset
        # with Spline without dropping the first feature,
        # and dropping the first feature after splitting X to train and test can cause unknown category error
        encoder_training = OneHotEncoder(sparse=False, drop='first')
        encoder_training.fit(X[cat_cols])
        cat_ohe_step = ('ohe', OneHotEncoder(sparse=False, drop='first', categories=encoder_training.categories_))

        cat_pipe = Pipeline([cat_ohe_step])
        num_pipe = Pipeline([('identity', FunctionTransformer())])
        transformers = [
            ('cat', cat_pipe, cat_cols),
            ('num', num_pipe, num_cols)
        ]
        ct = ColumnTransformer(transformers=transformers)

    base_model = params["base_model"]
    reg_l2 = params["reg_l2"]
    n_rounds = params["n_rounds"]
    early_stop = params["early_stop"]
    monotone_constraints = None

    if "monotone_constraints" in params:
        monotone_constraints = params["monotone_constraints"]

    if base_model == "XGB":
        pipe = Pipeline([
            ('ct', ct),#'monotone_constraints'] = (1,-1)
            (base_model, XGMModel(problem=problem, val_size=val_size, max_depth=5,
                                  subsample=params["subsample"], reg_l2=reg_l2,
                                  monotone_constraints=monotone_constraints, n_rounds=n_rounds, early_stop=early_stop))
        ])
        return pipe

    if base_model.lower() == "spline":
        prev_model = save_load_model(params['dataset_fun']()['dataset_name'], get_model_name(params), 0, model=None)
        mylam = None
        if prev_model is not None:
            mylam = prev_model[1].model.lam
        pipe = Pipeline([
            ('ct', ct),
            ("spline", GAMModel(problem=problem, val_size=val_size, mylam=mylam)),
        ])
        return pipe

    objective = 'reg:squarederror'
    if problem == "classification":
        objective = 'binary:logistic'
    elif problem == "survival":
        objective = "survival:cox"

    reg_l2_inter = params["reg_l2_inter"]
    interactions = params["interactions"]
    outer_bags = params["outer_bags"]
    feature_traverse = params["feature_traverse"]
    learning_rate = params["learning_rate"]

    inner_bags = 0
    subsample = 1
    if "inner_bags" in params:
        inner_bags = params["inner_bags"]
        subsample = 0.5

    # No pipeline needed due to EBM handling string datatypes
    if "EBM" in base_model:
        if problem == "classification":
            model = ExplainableBoostingClassifier(n_jobs=4, interactions=interactions, outer_bags=outer_bags,
                                                  validation_size=val_size, learning_rate=learning_rate,
                                                  random_state=RANDOM_STATE, max_rounds=n_rounds, inner_bags=inner_bags,
                                                  early_stopping_rounds=early_stop, max_bins=params['max_bins'],
                                                  max_interaction_bins=params['max_bins_interactions'])
        else:
            model = ExplainableBoostingRegressor(n_jobs=4, interactions=interactions, outer_bags=outer_bags,
                                                 validation_size=val_size, learning_rate=learning_rate,
                                                 random_state=RANDOM_STATE, max_rounds=n_rounds, inner_bags=inner_bags,
                                                 early_stopping_rounds=early_stop,max_bins=params['max_bins'],
                                                 max_interaction_bins=params['max_bins_interactions'],)
    else:
        model = egboost.ExplainableGBM(objective=objective, reg_l2=[reg_l2, reg_l2_inter], n_jobs=4,
                                         validation_size=val_size, interactions=interactions, max_leaves_inter=8,
                                         max_bins=params['max_bins'],
                                         max_interaction_bins=params['max_bins_interactions'],
                                         max_leaves=3, subsample=subsample, learning_rate=learning_rate,
                                         max_rounds=n_rounds, random_state=RANDOM_STATE, outer_bags=outer_bags,
                                         early_stopping_rounds=early_stop, inner_bags=inner_bags,
                                         monotone_constraints=monotone_constraints,
                                         feature_traverse=feature_traverse)

    return model


def c_statistic_harrell(pred, labels):
    total = 0
    matches = 0
    for i in range(len(labels)):
        for j in range(len(labels)):
            if labels[j] > 0 and abs(labels[i]) > labels[j]:
                total += 1
                if pred[j] > pred[i]:
                    matches += 1
    return matches / total


def eval_perf(model, X_test, y_test, problem):
    from sklearn.metrics import roc_auc_score

    if problem == 'classification':
        test_y_pred = model.predict_proba(X_test)
        if (len(test_y_pred.shape) > 1) and (test_y_pred.shape[1] > 1):
            test_y_pred = test_y_pred[:, 1]
        test_score = roc_auc_score(y_test, test_y_pred)
    elif 'regression' in problem:
        test_y_pred = model.predict(X_test)
        test_score = -1 * np.sqrt(((y_test - test_y_pred) ** 2).mean())
    elif problem == 'survival':  # survival.
        test_y_pred = model.predict(X_test)
        test_score = c_statistic_harrell(test_y_pred, np.array(y_test))

    return test_score, test_y_pred


def get_model_name(params):
    model_name = params["base_model"]

    if "EGB_XGB" in params["base_model"]:
        model_name += "_"+params["feature_traverse"][0].name+"_"+params["feature_traverse"][1].name
    return model_name


def get_accuracy(X_train, y_train, X_test, y_test, problem, d_name, split_idx, params):
    print(d_name, params, split_idx, end='\r')

    model_name = get_model_name(params)
    model = get_model(params, problem, pd.concat([X_train, X_test]))
    start_time = time.time()
    model.fit(X_train, y_train)
    record = {'fit_time': float(time.time() - start_time), 'fit_trees_main': 0, 'fit_trees_inter': 0}

    num_trees_main = num_trees_inter = []

    if 'EBM' in params['base_model']:
        num_inter = np.sum([1 for f in model.feature_groups_ if len(f) > 1])
        num_trees_main = [(model.main_episode_idx_+1) * X_train.shape[1]*params['inner_bags'] for model in model.bagged_models_]
        num_trees_inter = [(model.inter_episode_idx_+1) * num_inter * params['inner_bags'] for model in model.bagged_models_]
    elif "EGB" in params['base_model']:
        num_trees_main = [model.num_trees_main]
        num_trees_inter = [model.num_trees_inter]
        model.estimators = None

    save_load_model(d_name, model_name, split_idx, model=model)

    if len(num_trees_main) > 0:
        record['fit_trees_main'] = np.mean(num_trees_main)
    if len(num_trees_inter) > 0:
        record['fit_trees_inter'] = np.mean(num_trees_inter)

    record['test_score'], _ = eval_perf(model, X_test, y_test, problem)
    record['model_name'] = model_name

    return record


def predict_score_with_each_feature(model, X, add_std=False):
    from interpret.glassbox.ebm.utils import EBMUtils

    if isinstance(model, ExplainableBoostingRegressor) or isinstance(model, ExplainableBoostingClassifier):
        samples = model.preprocessor_.transform(X.values)
        samples = np.ascontiguousarray(samples.T)

        if model.interactions != 0:
            pair_samples = model.pair_preprocessor_.transform(X.values)
            pair_samples = np.ascontiguousarray(pair_samples.T)
        else:
            pair_samples = None

        scores_dic = {'offset': np.ones(X.values.shape[0]) * model.intercept_}
        scores_gen = EBMUtils.scores_by_feature_group(
            samples, pair_samples, model.feature_groups_, model.additive_terms_
        )
        for set_idx, _, scores in scores_gen:
            scores_dic[model.feature_names[set_idx]] = scores
        scores_df = pd.DataFrame(scores_dic)

        if add_std:
            std_dic = {}
            std_gen = EBMUtils.scores_by_feature_group(
                samples, pair_samples, model.feature_groups_, model.term_standard_deviations_
            )
            for set_idx, _, scores in std_gen:
                std_dic[model.feature_names[set_idx]] = scores
            std_df = pd.DataFrame(std_dic)

    elif isinstance(model, Pipeline):
        # spline . it use one hot encoder.
        # we need to extract the contribution of each feature and sum each one hot encoded columns
        # print("its a spline")
        pipeline = model
        ct_step = pipeline.named_steps['ct'].transformers_
        cat_columns = list(ct_step[0][2])
        num_columns = list(ct_step[1][2])
        final_cols_cat = []
        if len(cat_columns) > 0:
            final_cols_cat = list(ct_step[0][1].named_steps['ohe'].get_feature_names(cat_columns))
        # display(cat_columns)
        # display(final_cols_cat)
        # display(num_columns)
        final_columns = final_cols_cat + num_columns
        # display(final_columns)
        transformed = pipeline.named_steps['ct'].transform(X)
        transformed_df = pd.DataFrame(transformed, columns=final_columns)
        # display(transformed_df)

        gam_log = pipeline.named_steps["spline"].model

        # print(gam_log.coef_[-1])
        scores_with_dummies = transformed_df.copy()
        std_with_dummies = transformed_df.copy()
        for i, col in enumerate(final_columns):
            scores_with_dummies[col], std_with_dummies[col] = gam_log.partial_dependence(i, transformed, width=0.95)

        scores_with_dummies['offset'] = gam_log.coef_[-1]

        # display(exp_with_dummies)
        # display(sigmoid(scores_with_dummies.sum(axis=1)))

        for cat in cat_columns:
            cols_final_curr = [col for col in final_cols_cat if col.startswith(cat + "_")]
            scores_with_dummies[cat] = scores_with_dummies[cols_final_curr].sum(axis=1)
            scores_with_dummies = scores_with_dummies.drop(cols_final_curr, axis=1)

            std_with_dummies[cat] = std_with_dummies[cols_final_curr].sum(axis=1)
            std_with_dummies = std_with_dummies.drop(cols_final_curr, axis=1)

        scores_df = scores_with_dummies
        std_df = std_with_dummies

    else:
        if add_std:
            explain_local, explain_local_std = model.explain_local(X, add_std=add_std)
            std_df = pd.DataFrame(explain_local_std).rename({0: "offset"}, axis=1)
        else:
            explain_local = model.explain_local(X, add_std=add_std)
        scores_df = pd.DataFrame(explain_local).rename({0: "offset"}, axis=1)
    if add_std:
        return scores_df, std_df
    return scores_df


def sigmoid(x):
    "Numerically stable sigmoid function."
    return np.where(x >= 0,
                    1 / (1 + np.exp(-x)),
                    np.exp(x) / (1 + np.exp(x)))


def addExp(model, X, Y, problem):
    def eval_perf_sparseness(pred_score, y, problem):
        if problem == 'classification':
            pred_score = sigmoid(pred_score)
            eps = np.finfo(pred_score.dtype).eps  # 'logloss'
            p = np.clip(pred_score, eps, 1. - eps)
            return np.mean(y * -np.log(p) + (1. - y) * (-np.log(1. - p)))

        return np.sqrt(((y - pred_score) ** 2).mean())

    result = {'feat_names': ["offset"]}

    scores_report = scores_select = predict_score_with_each_feature(model, X)

    result['feat_perf'] = [eval_perf_sparseness(scores_report[result['feat_names']].values.sum(axis=1),
                                                Y, problem)]

    for _ in range(scores_select.shape[1] - 1):

        best_perf, best_feat = None, None
        for f_name in scores_select.columns:
            if f_name in result['feat_names']:
                continue
            the_perf = eval_perf_sparseness(scores_select[result['feat_names'] + [f_name]].values.sum(axis=1),
                                            Y, problem)

            if (best_perf is None) or (the_perf < best_perf):
                best_perf, best_feat = the_perf, f_name

        result['feat_names'] += [best_feat]
        result['feat_perf'] += [the_perf]
    return result


def get_feature_sparseness(X_train, y_train, X_test, y_test, problem, d_name, split_idx, params):
    model_name = get_model_name(params)

    model = save_load_model(d_name, model_name, split_idx)
    if model is None:
        print("no model %s" % model_name)
        return {}

    # X_test = X_test.reindex(sorted(X_test.columns), axis=1)# we need to make sure its in the same order for ohe
    record = addExp(model, X_test, y_test, problem)
    record['model_name'] = model_name

    return record


def save_feature_sparseness_summary():
    import ast

    def my_normalize(feat_perf):

        largest = feat_perf[0]  # Initial performance
        lowest = feat_perf[-1]  # Final performance
        result = (feat_perf - lowest) / (largest - lowest)
        return result

    def cal_area_till_zero(feat_perf):
        feat_perf = np.array(feat_perf)
        return np.trapz(feat_perf, dx=(1. / (len(feat_perf) - 1)))  # /np.std(feat_perf)

    def compute_summary(results):

        # print(results)
        results = results["feat_perf"]
        per_all = []
        for i, res in enumerate(results):
            perfs = np.array(list(res)).astype(float)
            per_all += [my_normalize(perfs)]
            # per_all += [perfs]
        the_mean = np.mean(per_all, axis=0)  # [1:]
        the_std = np.std(per_all, axis=0)  # [1:]
        for i in range(len(the_mean)):
            if i > 0 and the_mean[i] > the_mean[i - 1]:  # set monotonic constraints down
                the_mean[i] = the_mean[i - 1]

        # | (np.array(the_mean) > 1)
        max_index_on_zero_array = np.arange(len(the_mean))[(np.array(the_mean) < 0)]
        max_index_on_zero = len(the_mean)
        if len(max_index_on_zero_array) > 0:
            max_index_on_zero = max_index_on_zero_array[0]

        return pd.Series({"mean_perf": the_mean, "std_perf": the_std, "max_index_on_zero": max_index_on_zero})

    results_sparseness = pd.read_csv("feature_sparseness_results_splits.csv")

    results_sparseness = results_sparseness[results_sparseness["feat_perf"] != "0"]
    results_sparseness.feat_perf = results_sparseness.feat_perf.apply(ast.literal_eval)

    summary = results_sparseness.groupby(["d_name", "model_name"]).apply(compute_summary).reset_index()

    summary["interactions"] = summary.model_name.str.contains("interactions=16")
    min_max_index_on_zero = summary.groupby(["d_name", "interactions"]).max_index_on_zero.min().rename(
        "min_max_index_on_zero")
    summary = summary.merge(min_max_index_on_zero, on=["d_name", "interactions"])

    summary["mean_perf"] = summary.apply(lambda r: list(r["mean_perf"][:r["min_max_index_on_zero"]]), axis=1)
    summary["std_perf"] = summary.apply(lambda r: list(r["std_perf"][:r["min_max_index_on_zero"]]), axis=1)
    summary["area_till_zero_mean"] = summary.apply(lambda r: cal_area_till_zero(r["mean_perf"]), axis=1)
    summary.to_csv("feature_sparseness_results.csv")


def get_bias_var(X_train, y_train, X_test, y_test, problem, d_name, split_idx, params):
    all_sub_records = []
    model_name = ""
    total_fit_time = 0
    ss = get_shuffle_split(problem, N_SPLITS, RANDOM_STATE, test_size=(1. - SUBSAMPLE_RATIO))
    for sub_idx, (subsample_idxes, _) in enumerate(ss.split(X_train, y_train)):
        print(d_name, params, split_idx, sub_idx, end='\r')
        X, y = X_train.iloc[subsample_idxes], y_train.iloc[subsample_idxes]

        model = get_model(params, problem, pd.concat([X_train, X_test]))
        model_name = get_model_name(params)

        start_time = time.time()
        model.fit(X, y)
        total_fit_time += float(time.time() - start_time)

        score_pred = eval_perf(model, X_test, y_test, problem)
        all_sub_records.append({'test_score': score_pred[0],
                      'test_y_pred': score_pred[1],
                      'test_mse': ((y_test - score_pred[1]) ** 2).mean()
                      })

    sub_df = pd.DataFrame(all_sub_records)

    variance = np.mean(np.var(np.array(sub_df.test_y_pred), axis=0))
    avg_test_y_pred = np.mean(np.array(sub_df.test_y_pred), axis=0)
    bias = np.mean((avg_test_y_pred - y_test.values) ** 2)
    error = np.mean(sub_df.test_mse)
    assert np.isclose(bias + variance, error), 'bias: %f, var: %f, error: %f' % (bias, variance, error)

    record = {}
    record['model_name'] = model_name
    record['bias'] = bias
    record['variance'] = variance
    record['test_mse'] = error
    record['test_score'] = np.mean(sub_df.test_score)
    record['n_subsamples'] = N_SUBSAMPLES
    record['subsample_ratio'] = SUBSAMPLE_RATIO
    record['fit_time'] = total_fit_time

    return record


def update_best_hyperparameters(X_train, y_train, params, problem, dataset_name):
    hyper_file_name = "hyper_parameters"
    hyper_file = Path(hyper_file_name + ".csv")
    all_records_hyper_df = None
    if hyper_file.is_file():
        all_records_hyper_df = pd.read_csv(hyper_file)

    grid_hyper_dic = {"XGB": {'subsample': [0.5, 1], 'reg_l2': [1, 8]},
                      "EGB_XGB_GA2M": {'interactions': [8, 16, 32], 'reg_l2_inter': [1, 8]},
                      "EBM_GA2M": {'interactions': [8, 16, 32]}
                      }

    if params['base_model'] in grid_hyper_dic.keys():
        if all_records_hyper_df is not None:
            hyper_curr_df = all_records_hyper_df[(all_records_hyper_df["model_name"] == params['base_model']) &
                                                 (all_records_hyper_df["d_name"] == dataset_name)]

        if all_records_hyper_df is None or hyper_curr_df.shape[0] == 0:

            X_train_hyper, X_val_hyper, y_train_hyper, y_val_hyper = train_test_split(
                X_train, y_train, test_size=0.176  ## 85% * 0.176 = 15%
            )
            all_records_hyper = []
            grid_hyper = grid_hyper_dic[params['base_model']]

            for _, params_hyper in enumerate(ParameterGrid(grid_hyper)):
                if all_records_hyper_df is not None and \
                        all_records_hyper_df[(all_records_hyper_df["model_name"] == params['base_model']) &
                                             (all_records_hyper_df["d_name"] == dataset_name) &
                                             (all_records_hyper_df["params_hyper"] == str(params_hyper))].shape[0] > 0:
                    continue
                if 'interactions' in params_hyper and params_hyper['interactions'] > 8 and params_hyper[
                    'interactions'] > \
                        ((X_train_hyper.shape[1] * (X_train_hyper.shape[1] - 1)) / 2):  # waste of time
                    continue
                params.update(params_hyper)
                record_hyper = get_accuracy(X_train_hyper, y_train_hyper, X_val_hyper, y_val_hyper, problem,
                                            dataset_name,
                                            0, params)
                record_hyper.update(params_hyper)
                record_hyper["params_hyper"] = str(params_hyper)
                all_records_hyper += [record_hyper]
            hyper_curr_df = pd.DataFrame(all_records_hyper)
            hyper_curr_df['d_name'] = dataset_name
            hyper_curr_df["base_model"] = params['base_model']

            if all_records_hyper_df is not None:
                all_records_hyper_df = pd.concat([hyper_curr_df, all_records_hyper_df])
            else:
                all_records_hyper_df = hyper_curr_df
            all_records_hyper_df.to_csv(hyper_file, index=False)

            all_records_hyper_df.sort_values('test_score', ascending=False).drop_duplicates(["d_name", "model_name"]). \
                to_csv(hyper_file_name + "_best.csv", index=False)

        best_records_hyper_df = pd.read_csv(hyper_file_name + "_best.csv")
        best_params = best_records_hyper_df[(best_records_hyper_df["model_name"] == params['base_model']) &
                                            (best_records_hyper_df["d_name"] == dataset_name)]
        best_params = best_params.to_dict('records')[0]['params_hyper']
        import ast
        params.update(ast.literal_eval(best_params))


def cross_validate(file_name, func_eva, save_summary=True, only_gams=False, datasets_list=None, tune=False):
    splits_file = Path(file_name + "_splits" + ".csv")
    all_records = []
    all_records_df = None
    if splits_file.is_file():
        all_records_df = pd.read_csv(splits_file)
    if datasets_list is None:
        datasets_list = DATASETS_LIST

    E_BF = egboost.FeatureTraverse.BEST_FIT
    E_R = egboost.FeatureTraverse.RANDOM
    E_C = egboost.FeatureTraverse.CYCLIC
    E_CR = egboost.FeatureTraverse.CYCLIC_RANDOM
    E_CRE = egboost.FeatureTraverse.CYCLIC_REVERSE

    grid = ParameterGrid(
        {'dataset_fun': datasets_list, 'base_model': ["EGB_XGB_GAM", "EBM_GAM",  "EBM_GA2M", "EGB_XGB_GA2M", "XGB", "Spline"],
         'feature_traverse': [[E_CRE, E_CRE], [E_C, E_C], [E_CR, E_CR], [E_R, E_R], [E_C, E_BF], [E_BF, E_BF]],
         'n_features': [10, 100, 1000]
         })

    for grid_i, params in enumerate(grid):

        params['n_rounds'] = N_ROUNDS
        params['early_stop'] = EARLY_STOP
        params['learning_rate'] = 0.01
        params['outer_bags'] = 10
        params["inner_bags"] = 10
        params["reg_l2"] = 0
        params["reg_l2_inter"] = 4
        params["interactions"] = 0
        params["subsample"] = 1
        params['max_bins'] = 256
        params['max_bins_interactions'] = 32
        feature_traverse = params["feature_traverse"]

        if params['dataset_fun'] != gen_synth_data_ordering_exp:
            if feature_traverse not in [[E_C, E_C], [E_CRE, E_CRE], [E_BF, E_BF], [E_C, E_BF]]:
                continue
            if params["n_features"] > 10:
                continue
            dataset = params['dataset_fun']()
        else:
            if file_name == "bias_variance_results" or params["base_model"] not in ["EBM_GAM", "EGB_XGB_GAM"]:
                continue
            params['n_rounds'] = int(10000 / (params["n_features"]))#to reduce the training time and set same number of trees for all experiments
            params['max_bins'] = 63  # to reduce the training time
            if params["base_model"] == "EGB_XGB_GAM":
                params['early_stop'] = -1 #to reduce the training time

            dataset = params['dataset_fun'](params["n_features"])

        if "GA2M" in params["base_model"]:
            params["interactions"] = 16

        X, y = dataset['full']['X'], dataset['full']['y']

        problem = dataset['problem']
        if 'monotone_constraints' in dataset:
            params['monotone_constraints'] = dataset['monotone_constraints']
        if problem in ["survival", 'regression_monotone'] and (
                params["feature_traverse"] != [E_C, E_C] or ("XGB" not in params["base_model"])):
            continue

        if only_gams and params["base_model"] == "XGB":
            continue

        base_model = params["base_model"]
        model_name = get_model_name(params)
        dataset_name = dataset['dataset_name']

        if all_records_df is not None and (all_records_df[(all_records_df["model_name"] == model_name) &
                                                          (all_records_df["d_name"] == dataset_name)].shape[0] > 0):
            continue

        if ("GA2M" not in model_name) and (feature_traverse[0] != feature_traverse[1]):
            print("continue - interactions == 0")
            continue

        if ("EGB" not in base_model) and (feature_traverse != [E_C, E_C]):
            continue

        ss = get_shuffle_split(problem, N_SPLITS, RANDOM_STATE)
        print(params)

        for split_idx, (train_idx, test_idx) in enumerate(ss.split(X, y)):

            date = datetime.now()
            print("Hour %d:%d:%d, Split idx %d" % (date.hour, date.minute, date.second, split_idx))
            X_test = X.iloc[test_idx]
            y_test = y.iloc[test_idx]
            X_train = X.iloc[train_idx]
            y_train = y.iloc[train_idx]
            if tune:
                update_best_hyperparameters(X_train, y_train, params, problem, dataset_name)

            record = func_eva(X_train, y_train, X_test, y_test, problem, dataset_name, split_idx, params)

            record['d_name'] = dataset_name
            record['split_idx'] = split_idx
            record['n_splits'] = N_SPLITS
            record['random_state'] = RANDOM_STATE
            record['test_size'] = len(y_test)

            all_records += [record]

        if all_records_df is not None:
            all_records_df = pd.concat([pd.DataFrame(all_records), all_records_df]).drop_duplicates(
                subset=["d_name", "model_name", "split_idx"]).fillna(0)
        else:
            all_records_df = pd.DataFrame(all_records).fillna(0)
        all_records_df.to_csv(splits_file, index=False)
        if save_summary:
            grouped = all_records_df.drop("split_idx", axis=1).groupby(["d_name", "model_name"])
            grouped.mean().join(grouped.std(), lsuffix="_mean", rsuffix="_std").to_csv(file_name + ".csv")


RUN_DATASETS_DESC = True
RUN_ACCURACY = True
RUN_FEATURE_SPARSENESS = True
RUN_BIAS_VARIANCE = True
RUN_USECASES = True

DATASETS_LIST = [gen_synth_data_correlation_exp, gen_synth_data_ordering_exp,
                 load_breast_data, load_crimes_data, load_fico_score_data,
                 load_california_housing_data, load_compas_data, load_support2_data, load_bike_sharing_data,
                 load_telco_churn_data, load_wine_data, load_adult_data]
if __name__ == "__main__":

    if RUN_DATASETS_DESC:
        datasets_desc = []
        for dataset_fun in DATASETS_LIST + [load_nhanesi_data]:
            dataset = dataset_fun()
            del dataset['full']
            datasets_desc += [dataset]
        pd.DataFrame(datasets_desc).to_csv("datasets_desc.csv")

    if RUN_FEATURE_SPARSENESS:
        cross_validate("feature_sparseness_results", get_feature_sparseness, save_summary=False, only_gams=True)
        save_feature_sparseness_summary()

    if RUN_ACCURACY:
        cross_validate("accuracy_results", get_accuracy)

    if RUN_FEATURE_SPARSENESS:
        cross_validate("feature_sparseness_results", get_feature_sparseness, save_summary=False, only_gams=True)
        save_feature_sparseness_summary()

    if RUN_USECASES:
        cross_validate("accuracy_results_usecases", get_accuracy,
                       datasets_list=[load_california_housing_data_monotone, load_nhanesi_data])

    if RUN_BIAS_VARIANCE:
        cross_validate("bias_variance_results", get_bias_var, tune=False)
