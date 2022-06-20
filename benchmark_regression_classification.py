import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_validate, ShuffleSplit, StratifiedShuffleSplit

from interpret.glassbox import ExplainableBoostingRegressor, ExplainableBoostingClassifier
import sys

sys.path.insert(0, '/home/avi/git/distributions/LightGBM/python-package')
sys.path.insert(0, '/home/avi/git/distributions/xgboost/python-package')
import lightgbm as lgb
import xgboost as xgb
import egbm_pure_custom as egbm_pure
import pickle

RANDOM_STATE = 42
N_SPLITS = 5
N_ROUNDS = 30000
EARLY_STOP = 50


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

    return dataset


def load_synthetic_class_data():
    from sklearn.datasets import make_classification

    X, y = make_classification(n_samples=10000, n_features=4, n_informative=4, n_redundant=0, n_repeated=0, n_classes=2,
                               n_clusters_per_class=4, class_sep=1, flip_y=0, random_state=17)
    dataset = {
        'dataset_name': 'synthetic',
        'problem': 'classification',
        'full': {
            'X': pd.DataFrame(X),
            'y': pd.Series(y),
        },
    }
    return add_dataset_stats(dataset)


def load_compas_data():
    # COMPAS: https://www.kaggle.com/danofer/compass
    df = pd.read_csv(r'data/uci/propublica_data_for_fairml.csv')
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
        "data/uci/adult.data",
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


def load_heart_data():
    # https://www.kaggle.com/sonumj/heart-disease-dataset-from-uci
    df = pd.read_csv(r'data/uci/heart_disease_dataset_UCI.csv')
    train_cols = df.columns[0:-1]
    label = df.columns[-1]
    X_df = df[train_cols]
    y_df = df[label]
    dataset = {
        'dataset_name': 'heart',
        'problem': 'classification',
        'full': {
            'X': X_df,
            'y': y_df,
        },
    }

    return add_dataset_stats(dataset)


def load_credit_data():
    # https://www.kaggle.com/mlg-ulb/creditcardfraud
    df = pd.read_csv(r'data/uci/creditcard.tar.xz')
    train_cols = df.columns[0:-1]
    label = df.columns[-1]
    X_df = df[train_cols]
    y_df = df[label]
    dataset = {
        'dataset_name': 'credit',
        'problem': 'classification',
        'full': {
            'X': X_df,
            'y': y_df,
        },
    }

    return add_dataset_stats(dataset)


def load_telco_churn_data():
    # https://www.kaggle.com/blastchar/telco-customer-churn/downloads/WA_Fn-UseC_-Telco-Customer-Churn.csv/1
    df = pd.read_csv(r'data/surv/telco_churn.csv')
    # small number of the values are not recognized as numbers, so binning it will give us all the unique values (6532)
    # It increase the running of learning interactions in EBM by alot (10 sec before fixing, 1 sec after)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    print("before dropping na %d" % df.shape[0])
    df = df.dropna()
    print("after dropping na %d" % df.shape[0])

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


def load_fico_data():
    df = pd.read_csv(r'data/fico.csv')
    train_cols = df.columns[:-1]  # First column is an ID
    label = df.columns[-1]
    X_df = df[train_cols]
    y_df = df[label]
    dataset = {
        'dataset_name': 'fico',
        'problem': 'classification',
        'full': {
            'X': X_df,
            'y': y_df,
        },
    }

    return add_dataset_stats(dataset)


def load_support2_data():
    X_df = pd.read_csv(r'data/support2/x_new.csv')
    y_df = pd.read_csv(r'data/support2/y.csv').reset_index()['hospdead']
    dataset = {
        'dataset_name': 'support2',
        'problem': 'classification',
        'full': {
            'X': X_df,
            'y': y_df,
        },
    }

    return add_dataset_stats(dataset)


def load_credit_default_taiwan_data():

    meta_info = {'LIMIT_BAL':{'type':'continuous'},
                 'PAY_0':{'type':'continuous'},
                 'PAY_2':{'type':'continuous'},
                 'PAY_3':{'type':'continuous'},
                 'PAY_4':{'type':'continuous'},
                 'PAY_5':{'type':'continuous'},
                 'PAY_6':{'type':'continuous'},
                 'BILL_AMT1':{'type':'continuous'},
                 'BILL_AMT2':{'type':'continuous'},
                 'BILL_AMT3':{'type':'continuous'},
                 'BILL_AMT4':{'type':'continuous'},
                 'BILL_AMT5':{'type':'continuous'},
                 'BILL_AMT6':{'type':'continuous'},
                 'PAY_AMT1':{'type':'continuous'},
                 'PAY_AMT2':{'type':'continuous'},
                 'PAY_AMT3':{'type':'continuous'},
                 'PAY_AMT456':{'type':'continuous'},
                 'FLAG_UTIL_RAT1':{'type':'categorical'},
                 'UTIL_RAT1':{'type':'continuous'},
                 'UTIL_RAT_AVG':{'type':'continuous'},
                 'UTIL_RAT_RANGE':{'type':'continuous'},
                 'UTIL_RAT_MAX':{'type':'continuous'},
                 'FLAG_PAY_RAT1':{'type':'categorical'},
                 'PAY_RAT1':{'type':'continuous'},
                 'PAY_RAT_AVG':{'type':'continuous'},
                 'PAY_RAT_RANGE':{'type':'continuous'},
                 'PAY_RAT_MAX':{'type':'continuous'},
                 'Default Payment':{'type':'target'}}

    data = pd.read_csv('data/credit_data_processed.csv', index_col=[0])
    X_df, y_df = data.loc[:,list(meta_info.keys())[:-1]], data['default.payment.next.month']

    dataset = {
        'dataset_name': 'credit_default_taiwan',
        'problem': 'classification',
        'full': {
            'X': X_df,
            'y': y_df,
        },
    }
    return add_dataset_stats(dataset)


def load_parkinsons_data():
    df = pd.read_csv("data/uci/parkinsons.csv")
    train_cols = df.columns[0:-2]
    label = df.columns[-2]
    X_df = df[train_cols]
    y_df = df[label]
    dataset = {
        'dataset_name': 'parkinsons',
        'problem': 'regression',
        'full': {
            'X': X_df,
            'y': y_df,
        },
    }
    return add_dataset_stats(dataset)


def load_electrical_grid_data():
    df = pd.read_csv("data/uci/electrical_grid.csv")
    train_cols = df.columns[0:-2]
    label = df.columns[-2]
    X_df = df[train_cols]
    y_df = df[label]
    dataset = {
        'dataset_name': 'electrical_grid',
        'problem': 'regression',
        'full': {
            'X': X_df,
            'y': y_df,
        },
    }
    return add_dataset_stats(dataset)


def load_airfoil_data():
    df = pd.read_csv("data/uci/airfoil_self_noise.dat", sep="\t",
        header=None)
    train_cols = df.columns[0:-1]
    label = df.columns[-1]
    X_df = df[train_cols].astype(float)
    y_df = df[label].astype(float)
    dataset = {
        'dataset_name': 'airfoil',
        'problem': 'regression',
        'full': {
            'X': X_df,
            'y': y_df,
        },
    }
    return add_dataset_stats(dataset)


def load_elevators_data():
    df = pd.read_csv("data/uci/elevators.csv", header=None)
    print(df.isnull().sum())
    train_cols = df.columns[0:-1]
    label = df.columns[-1]
    X_df = df[train_cols].astype(float)
    y_df = df[label].astype(float)
    dataset = {
        'dataset_name': 'abalone',
        'problem': 'regression',
        'full': {
            'X': X_df,
            'y': y_df,
        },
    }
    return add_dataset_stats(dataset)


def load_synthetic_reg_data():
    from utils_data import gen_synth_data_correlation_exp

    X, y, _, _ = gen_synth_data_correlation_exp(noise_c=0.01, n=1000, noise_f=0, add_inter=True)

    dataset = {
        'dataset_name': 'synthetic_regression',
        'problem': 'regression',
        'full': {
            'X': X,
            'y': pd.Series(y),
        },
    }
    return add_dataset_stats(dataset)


def load_bike_sharing_data():
    data = pd.read_csv("data/uci/Bike-Sharing-Dataset/hour.csv").drop(["instant", "dteday", "casual", "registered"],
                                                                      axis=1)
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
        'MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup',
        'Latitude', 'Longitude'
    ]

    df = pd.read_csv("data/cal_housing.data")
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


def load_concrete_data():
    data = pd.read_excel(
        "https://archive.ics.uci.edu/ml/machine-learning-databases/concrete/compressive/Concrete_Data.xls"
    )
    X, y = data.iloc[:, :-1], data.iloc[:, -1]
    dataset = {
        'dataset_name': 'concrete',
        'problem': 'regression',
        'full': {
            'X': X,
            'y': y,
        },
    }
    return add_dataset_stats(dataset)


def load_power_data():
    data = pd.read_excel("data/uci/power-plant.xlsx")
    X, y = data.iloc[:, :-1], data.iloc[:, -1]
    dataset = {
        'dataset_name': 'power',
        'problem': 'regression',
        'full': {
            'X': X,
            'y': y,
        },
    }
    return add_dataset_stats(dataset)


def format_n(x):
    return "{0:.3f}".format(x)


def save_load_model(dataset, name, split_idx, model=None):

    import os
    output_dir = "models/"
    model_path = os.path.join(output_dir, '%s_%s_%d.pkl' % (dataset, name, split_idx))
    if model is not None:
        pickle.dump(model, open(model_path, 'wb'))
    else:
        try:
            with open(model_path, 'rb') as fp:
                return pickle.load(fp)
        except:
            return None


def get_shuffle_split(problem, n_splits, random_state):

    # Evaluate model
    if problem == 'classification':
        ss = StratifiedShuffleSplit(n_splits=n_splits, test_size=0.15, random_state=random_state)
        scoring = 'roc_auc'
    else:
        ss = ShuffleSplit(n_splits=n_splits, test_size=0.15, random_state=random_state)
        scoring = 'neg_root_mean_squared_error'

    return ss, scoring


def process_model(clf, name, dataset, n_splits=5, random_state=42):
    X, y = dataset['full']['X'], dataset['full']['y']
    problem = dataset['problem']

    ss, scoring = get_shuffle_split(problem, n_splits, random_state)
    scores = cross_validate(
        clf, X, y, scoring=scoring, cv=ss,
        n_jobs=None, return_estimator=True
    )

    record = dict()
    record['model_name'] = name
    record['fit_time_mean'] = format_n(np.mean(scores['fit_time']))
    record['fit_time_std'] = format_n(np.std(scores['fit_time']))
    record['test_score_mean'] = format_n(np.mean(scores['test_score']))
    record['test_score_std'] = format_n(np.std(scores['test_score']))
    num_trees_main = []
    num_trees_inter = []

    if isinstance(clf, ExplainableBoostingRegressor) or isinstance(clf, ExplainableBoostingClassifier):
        print("ExplainableBoosting instance")
        for split_idx, est in enumerate(scores['estimator']):
            num_inter = np.sum([1 for f in est.feature_groups_ if len(f) > 1])
            num_trees_main += [model.main_episode_idx_ * X.shape[1] for model in est.bagged_models_]
            num_trees_inter += [model.inter_episode_idx_ * num_inter for model in est.bagged_models_]
            save_load_model(dataset['dataset_name'], name, split_idx, model=est)

    elif isinstance(clf, egbm_pure.ExplainableGBM):
        print("ExplainableBoosting instance")
        for split_idx, est in enumerate(scores['estimator']):
            num_trees_main += [est.num_trees_main]
            num_trees_inter += [est.num_trees_inter]
            est.estimators = None
            save_load_model(dataset['dataset_name'], name, split_idx, model=est)

    record['fit_trees_main'] = record['fit_trees_inter'] = 0
    if len(num_trees_main) > 0:
        record['fit_trees_main'] = format_n(np.mean(num_trees_main))
        record['fit_trees_inter'] = format_n(np.mean(num_trees_inter))

    return record


def benchmark_models(dataset, ct=None, n_splits=3, run_spline=True, random_state=42):
    X, y = dataset['full']['X'], dataset['full']['y']
    dataset_name = dataset['dataset_name']
    problem = dataset['problem']

    if ct is None:
        is_cat = np.array([dt.kind == 'O' for dt in X.dtypes])
        cat_cols = X.columns.values[is_cat]
        num_cols = X.columns.values[~is_cat]

        cat_ohe_step = ('ohe', OneHotEncoder(sparse=False,
                                             handle_unknown='ignore'))

        cat_pipe = Pipeline([cat_ohe_step])
        num_pipe = Pipeline([('identity', FunctionTransformer())])
        transformers = [
            ('cat', cat_pipe, cat_cols),
            ('num', num_pipe, num_cols)
        ]
        ct = ColumnTransformer(transformers=transformers)

    records = []

    summary_record = {}
    for key, val in dataset.items():
        if key != 'full':
            summary_record[key] = dataset[key]
    print()
    print('-' * 78)
    print(dataset_name)
    print('-' * 78)
    print(summary_record)
    print()

    from pygam import LogisticGAM, LinearGAM
    from sklearn.base import BaseEstimator
    from sklearn.model_selection import train_test_split

    class GAMModel(BaseEstimator):
        def __init__(
                self,
                problem
        ):
            self.classes_ = 1
            self.model = None
            self.problem = problem
            if self.problem == "classification":
                self.classes_ = 2

        def fit(
                self,
                X,
                y,
        ):
            if self.problem == "classification":
                self.model = LogisticGAM(max_iter=500, n_splines=10)
            else:
                self.model = LinearGAM(max_iter=500, n_splines=50)
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
                self, problem
        ):
            self.model = None
            self.problem = problem
            self.classes_ = 1
            if self.problem == "classification":
                self.classes_ = 2

        def fit(
                self,
                X,
                y,
        ):
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=0.15
            )
            params = {'eta': 0.1,
                      'tree_method': 'hist',
                      'verbosity': 0,
                      'grow_policy': 'lossguide',
                      'random_state': random_state
                      }
            if self.problem == "classification":
                params['objective'] = 'binary:logistic'

            train = xgb.DMatrix(X_train, label=y_train)
            val = xgb.DMatrix(X_val, label=y_val)

            early_stopping_tolerance = 1e-4
            early_stop = xgb.callback.EarlyStopping(rounds=EARLY_STOP,
                                                    min_delta=early_stopping_tolerance)

            self.model = xgb.train(params, train, N_ROUNDS, evals=[(val, 'eval')],
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

    class LGMModel(BaseEstimator):
        def __init__(
                self,
                problem
        ):
            self.classes_ = 1
            self.model = None
            self.problem = problem
            if self.problem == "classification":
                self.classes_ = 2

        def fit(
                self,
                X,
                y,
        ):
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=0.15
            )
            early_stopping_tolerance = 1e-4
            params = {
                'verbosity': -1,
                'learning_rate': 0.1,
                'random_state': random_state
            }
            if self.problem == "classification":
                params['objective'] = 'binary'
            eval_callback = lgb.callback.early_stopping(EARLY_STOP, True,
                                                        verbose=False,
                                                        min_delta=early_stopping_tolerance)
            train = lgb.Dataset(X_train, y_train, params=params)
            val = lgb.Dataset(X_val, y_val, params=params)

            self.model = lgb.train(params,
                                   train,
                                   num_boost_round=N_ROUNDS,
                                   valid_sets=[val],
                                   callbacks=[eval_callback])

            return self

        def predict(self, X):
            return self.model.predict(X)

        def decision_function(self, X):
            return self.model.predict(X)

        def predict_proba(self, X):
            return self.model.predict(X)

    pipe = Pipeline([
        ('ct', ct),
        ('xgb', XGMModel(problem=problem)),
    ])
    record = process_model(pipe, 'xgb', dataset=dataset, n_splits=n_splits, random_state=random_state)
    print(record)
    record.update(summary_record)
    records.append(record)

    pipe = Pipeline([
        ('ct', ct),
        ('lgb', LGMModel(problem=problem)),
    ])
    record = process_model(pipe, 'lgb', dataset=dataset, n_splits=n_splits)
    print(record)
    record.update(summary_record)
    records.append(record)

    from sklearn.model_selection import ParameterGrid

    objective = 'reg:squarederror'
    if problem == "classification":
        objective = 'binary:logistic'
    grid = ParameterGrid({'base_model': ["XGBoost", "ebm"], 'reg_l1': [[0, 0]], 'interactions': [0, 10],
                          'reg_l2': [[1, 4]],  'num_leaves_inter': [9], 'num_leaves': [3], 'subsample': [1],
                          'early': [50], 'outer_bags': [100], 'feature_traverse_random': [[True, True], [False, False], [True, False]]
                          })

    for grid_i, params in enumerate(grid):
        print("%d out of %d" % (grid_i + 1, len(grid)))
        print(params)

        reg_l1 = params["reg_l1"]
        reg_l2 = params["reg_l2"]
        num_leaves_inter = params["num_leaves_inter"]
        base_model = params["base_model"]
        interactions = params["interactions"]
        outer_bags = params["outer_bags"]
        feature_traverse_random = params["feature_traverse_random"]
        num_leaves = params["num_leaves"]
        early = params["early"]

        if interactions == 0 and feature_traverse_random == [True, False]:
            continue

        # No pipeline needed due to EBM handling string datatypes
        if base_model == "ebm":
            if feature_traverse_random != [True, True]:
                continue
            if problem == "classification":
                ebm_model = ExplainableBoostingClassifier(n_jobs=50, interactions=interactions, outer_bags=outer_bags, inner_bags=100, 
                                                          random_state=random_state, max_rounds=N_ROUNDS,
                                                          early_stopping_rounds=early)
            else:
                ebm_model = ExplainableBoostingRegressor(n_jobs=50, interactions=interactions, outer_bags=outer_bags, inner_bags=100,
                                                         random_state=random_state, max_rounds=N_ROUNDS,
                                                         early_stopping_rounds=early)
        else:
            ebm_model = egbm_pure.ExplainableGBM(objective=objective, reg_l1=reg_l1, reg_l2=reg_l2, n_jobs=8,
                                                 LightGBM_base_model=base_model == "LightGBM",
                                                 interactions=interactions, num_leaves_inter=num_leaves_inter, num_leaves=num_leaves,
                                                 max_rounds=N_ROUNDS, random_state=random_state, outer_bags=outer_bags,
                                                 early_stopping_rounds=early, feature_traverse_random=feature_traverse_random)
            # continue_learning=continue_learning
            ebm_model.fit(X, y)

        name = base_model
        name += "_rand="+str(feature_traverse_random)+"_interactions="+str(interactions)

        record = process_model(ebm_model, name, dataset=dataset, n_splits=n_splits, random_state=random_state)
        record["params"] = str(params)
        print(record)
        record.update(summary_record)
        records.append(record)

    if run_spline:
        pipe = Pipeline([
            ('ct', ct),
            ('spline', GAMModel(problem=problem)),
        ])
        record = process_model(pipe, 'spline', dataset=dataset, n_splits=n_splits, random_state=random_state)
        print(record)
        record.update(summary_record)
        records.append(record)

    return records


if __name__ == "__main__":
    results = []#load_concrete_data, load_synthetic_reg_data,
    #load_heart_data, load_telco_churn_data, load_credit_data,
    #                    load_compas_data, load_synthetic_class_data, load_adult_data, load_bike_sharing_data, load_concrete_data,
    #                    load_power_data, load_california_housing_data, load_fico_data, load_support2_data
    for dataset_fun in [load_heart_data, load_bike_sharing_data, load_power_data, load_california_housing_data, load_telco_churn_data,
                        load_compas_data, load_synthetic_class_data, load_adult_data,
                        load_fico_data, load_support2_data, load_credit_data]:#[load_credit_default_taiwan_data]:#[load_fico_data]:#[load_elevators_data]:#[load_airfoil_data]:#[load_electrical_grid_data]:#[load_california_housing_data]:
        dataset = dataset_fun()
        n_splits = N_SPLITS
        run_spline = True
        if dataset['dataset_name'] == 'credit':
            run_spline = False
        result = benchmark_models(dataset=dataset, run_spline=run_spline, n_splits=n_splits, random_state=RANDOM_STATE)
        print(pd.DataFrame(result).set_index("model_name").drop(["dataset_name"], axis=1))
        results += result

        df_results = pd.DataFrame(results)
        df_results.to_csv("results_regression8.csv")
        # df_results.to_csv("results_classification.csv")
