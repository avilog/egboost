import numpy as np
import pandas as pd
import scipy
from interpret.glassbox import ExplainableBoostingRegressor
from sklearn.model_selection import train_test_split
np.random.seed(42)


def get_robot():
    robot = np.array([
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 1, 0, 0, 1, 1, 0, 0],
        [0, 0, 1, 1, 0, 0, 1, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 1, 1, 1, 0, 0, 0],
        [0, 0, 0, 1, 1, 1, 1, 0, 0, 0],
        [0, 0, 0, 1, 1, 1, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ])
    return np.rot90(robot, 3)


def map_vals_shape(vals, shape):

    vals_bined = (10 * vals).astype(int)
    #vals_bined[:, 0] = 9 - vals_bined[:, 0]
    return shape[vals_bined[:, 0], vals_bined[:, 1]].astype(float)


def gen_synth_data_correlation_exp(noise_c=0.01, n=1000, noise_f=0, add_inter=False):

    x_to_loc = {"x1": lambda x: 0.5*np.sin(np.multiply(x, 9))}
    X_df = pd.DataFrame({"x" + str(i + 1): scipy.stats.uniform.rvs(size=n) for i in range(3+noise_f)})
    X_df["x2"] = X_df["x1"]

    x_inter_loc = 0
    if add_inter:
        x_inter_loc = X_df["x3"] * X_df["x1"]
    loc = (sum([x_to_loc[name](X_df[name]) for name in x_to_loc.keys()]) + x_inter_loc)
    print("mean" + str(loc.mean()))
    noise = np.random.normal(scale=noise_c, size=n)
    y = loc + noise
    return X_df, y, x_to_loc, noise


def gen_synth_data(noise_c=0.01, add_inter=False, n=5000, noise_f=0, random_state_add=True):

    x_to_loc = {"x1": lambda x: 0.5*np.sin(np.multiply(x, 9))}#  "x2": lambda x: np.exp(x) + ((np.array(x) > 0.5) & (np.array(x) < 0.52)
    if random_state_add:
        X_df = pd.DataFrame({"x" + str(i + 1): scipy.stats.uniform.rvs(size=n, random_state=i+1) for i in range(5+noise_f)})
    else:
        X_df = pd.DataFrame({"x" + str(i + 1): scipy.stats.uniform.rvs(size=n) for i in range(5+noise_f)})

    x_inter_loc = 0
    if add_inter:#9*X_df["x4"]*X_df["x1"]) + np.sin(X_df["x4"]*X_df["x2"])   map_vals_shape(X_df[["x2", "x6"]].values, get_robot()
        x_inter_loc = X_df["x4"]*X_df["x3"] + 0.5*np.sin(9*X_df["x2"]*X_df["x5"])#5*((X_df["x2"] > X_df["x3"]) &(X_df["x3"] > 0.2) & (X_df["x3"] < 0.8)).astype(float)
        #x_inter_scale = np.abs(X_df["x1"] - X_df["x4"]) + ((X_df["x2"] > X_df["x5"]) & ((X_df["x5"] > 0.2) & (X_df["x5"] < 0.8))).astype(float) #4*((X_df["x1"] > X_df["x4"]) & (X_df["x4"] < 0.5).astype(float) + ((X_df["x2"] > 0.5) & (X_df["x3"] < 0.8)).astype(float))
        #x_inter_scale = np.exp(np.sin(9*X_df["x4"]*X_df["x5"])+np.power(X_df["x1"]*X_df["x2"], 0.5))#4*((X_df["x1"] > X_df["x4"]) & (X_df["x4"] < 0.5).astype(float) + ((X_df["x2"] > 0.5) & (X_df["x3"] < 0.8)).astype(float))

    loc = (sum([x_to_loc[name](X_df[name]) for name in x_to_loc.keys()]) + x_inter_loc)
    print("mean" + str(loc.mean()))
    noise = np.random.normal(scale=noise_c, size=n)
    y = loc + noise
    return X_df, y, x_to_loc, noise


def gen_synth_data_back(noise_c=1, add_inter=False, n=5000):

    x_to_loc = {"x1": lambda x: np.sin(np.multiply(x, 9))}#  "x2": lambda x: np.exp(x)

    X_df = pd.DataFrame({"x" + str(i + 1): scipy.stats.uniform.rvs(size=n, random_state=i+1) for i in range(8)})
    X_df["x5"] = X_df["x4"]
    x_inter_loc = 0
    if add_inter:#9*X_df["x4"]*X_df["x1"]) + np.sin(X_df["x4"]*X_df["x2"])
        x_inter_loc = 6*X_df["x4"]*X_df["x3"] + map_vals_shape(X_df[["x2", "x6"]].values, get_robot())#5*((X_df["x2"] > X_df["x3"]) &(X_df["x3"] > 0.2) & (X_df["x3"] < 0.8)).astype(float)
        #x_inter_scale = np.abs(X_df["x1"] - X_df["x4"]) + ((X_df["x2"] > X_df["x5"]) & ((X_df["x5"] > 0.2) & (X_df["x5"] < 0.8))).astype(float) #4*((X_df["x1"] > X_df["x4"]) & (X_df["x4"] < 0.5).astype(float) + ((X_df["x2"] > 0.5) & (X_df["x3"] < 0.8)).astype(float))
        #x_inter_scale = np.exp(np.sin(9*X_df["x4"]*X_df["x5"])+np.power(X_df["x1"]*X_df["x2"], 0.5))#4*((X_df["x1"] > X_df["x4"]) & (X_df["x4"] < 0.5).astype(float) + ((X_df["x2"] > 0.5) & (X_df["x3"] < 0.8)).astype(float))

    loc = (sum([x_to_loc[name](X_df[name]) for name in x_to_loc.keys()]) + x_inter_loc)
    print("mean" + str(loc.mean()))
    y = scipy.stats.norm.rvs(loc=loc, scale=noise_c, size=n, random_state=42)
    return X_df, y, x_to_loc


def get_shape_functions_mean(add_inter_range=[True],noise_c_range=[1]):

    from sklearn.metrics import mean_squared_error
    import lightgam

    shapes_all = []
    stats_all = []

    #np.sqrt(mean_squared_error(y_test, preds))
    for add_inter in add_inter_range:
        for noise_c in noise_c_range:
            print("inter=" + str(add_inter) + " noise_c=" + str(noise_c))
            X, Y, x_to_loc, _ = gen_synth_data(noise_c=noise_c, add_inter=add_inter)
            X_train, X_test, y_train, y_test = train_test_split(
                X, Y, test_size=0.2, random_state=42)

            stats = {"noise_c": noise_c, "inter": add_inter}

            ebm = ExplainableBoostingRegressor(n_jobs=-1, interactions=5)
            ebm.fit(X_train, y_train)
            stats["rmse_ebm"] = np.sqrt(mean_squared_error(y_test, ebm.predict(X_test)))

            # explain with EBM start
            import egbm_pure_custom as egbm_pure

            egbm = egbm_pure.ExplainableGBM(interactions=5, LightGBM_base_model=False)
            egbm.fit(X_train, y_train)
            stats["rmse_egbm"] = np.sqrt(mean_squared_error(y_test, egbm.predict(X_test)))
            shapes = {}

            for feature_group_index, feature_indexes in enumerate(ebm.feature_groups_):
                if len(feature_indexes) == 1:
                    bin_labels = ebm.preprocessor_._get_bin_labels(feature_indexes[0])
                    name = ebm.feature_names[feature_indexes[0]]
                    shapes[name + "_ebm_bins"] = bin_labels[1:]
                    shapes[name + "_ebm"] = ebm.additive_terms_[feature_group_index][1:]
                    shapes[name + "_egbm"] = egbm.outputs_[feature_group_index][1:]
                    print(len(shapes[name + "_ebm_bins"]), len(shapes[name + "_ebm"]), len(shapes[name + "_egbm"]))


            # get original values
            for name in x_to_loc.keys():
                shapes[name + "_y"] = x_to_loc[name](shapes[name + "_ebm_bins"])
                shapes[name + "_y"] -= np.mean(shapes[name + "_y"])

            for key in shapes.keys():
                print(key)
                print(len(shapes[key]))

            shapes["noise_c"] = noise_c
            shapes["inter"] = add_inter
            if len(shapes_all) == 0:
                shapes_all = pd.DataFrame(shapes)
            else:
                shapes_all = pd.concat([shapes_all, pd.DataFrame(shapes)])
            stats_all += [stats]

    return pd.DataFrame(shapes_all), pd.DataFrame(stats_all)


def evaluate_shape_functions(shapes_all, x_range=range(1, 6)):
    curr_mean = []

    #shapes_all["x6_y"] = shapes_all["x2_y"]
    for (noise_c, add_inter), curre_shapes in shapes_all.groupby(["noise_c", "inter"]):
        print("noise_c " + str(noise_c) + " inter " + str(add_inter))
        for i in x_range:
                name = "x" + str(i)
                if name + "_y" not in curre_shapes.columns:
                    continue
                # display(curre_shapes)
                # algos =  [name+"_ngb",name+"_ebm_dist",name+"_ebm_dist_inter", name+"_y"]
                algos = [name + "_ebm", name + "_lgm_gam",name + "_xgb_gam", name + "_y"]

                curre_shapes_i = curre_shapes.set_index("x" + str(i) + "_ebm_bins")[algos]
                currs = curre_shapes_i.corr()[name + "_y"]
                print(currs)
                curr_mean += [{"noise_c": noise_c, "inter": add_inter,
                               "ebm": currs[name + "_ebm"], "lgm_gam": currs[name + "_lgm_gam"], "xgb_gam": currs[name + "_xgb_gam"],
                               "x": name}]
                curre_shapes_i.plot()

    return pd.DataFrame(curr_mean)


def save_results():

    import seaborn as sns
    import utils_data

    def get_sin():
        msin = np.zeros((10, 10))
        for i in range(10):
            for j in range(10):
                msin[i, j] = np.sin(i * j * 9 / 100)
        return msin

    ebm = stats_all[0]['ebm_inter']
    ebm_dist = stats_all[0]['ebm_dist_inter']
    ebm_dist_unpure = stats_all[0]['ebm_dist_inter_unpure']

    def print_heatmap(data, var1="X1", var2="X2", ismean=True, algo="",
                      xlabels=None, ylabels=None, xticks=None, yticks=None):
        import math

        df = pd.DataFrame(data.T)  # [1:,1:]
        df = df.rename_axis(index=var1, columns=var2)
        ax = sns.heatmap(df, cmap="plasma", cbar=True)
        if xlabels is None:
            xlabels = ylabels = [0, 0.5]
            xticks = yticks = [0, 5]

        ax.set_xticks(xticks)
        ax.set_yticks(yticks)
        ax.set_xticklabels(xlabels)
        ax.set_yticklabels(ylabels)
        ax.invert_yaxis()
        name = var1 + " vs. " + var2
        ax = ax.set_title(name)
        if ismean:
            name += "Mean "
        else:
            name += "Std "
        base_path = "results_explantions/synthetic/"
        figure_ = ax.get_figure()

        figure_.savefig(base_path + name + "_" + algo + ".png", dpi=400)
        plt.show()

    def print_heatmap_inter(ebm, param_i, inter_number, is_engb=True, pure=True):
        if is_engb:
            feature_indexes = ebm.interactions_chosen[param_i][inter_number]
            model_graph = ebm.outputs_inter_[param_i][inter_number]
            names = ebm.feature_names_[param_i]
            algo = "ENGB"
            if not pure:
                algo += "_UNPURE"
        else:
            feature_indexes = ebm.feature_groups_[inter_number]
            model_graph = ebm.additive_terms_[inter_number]
            names = ebm.feature_names
            algo = "EBM"

        model_graph = model_graph[1:, 1:]
        bin_labels_left = ebm.preprocessor_._get_bin_labels(feature_indexes[0])
        bin_labels_right = ebm.preprocessor_._get_bin_labels(feature_indexes[1])

        xlabels = [0, float(int(bin_labels_left[32] * 10)) / 10]
        ylabels = [0, float(int(bin_labels_right[32] * 10)) / 10]
        xticks = yticks = [0, 32]
        print_heatmap(model_graph, var1=names[feature_indexes[1]], var2=names[feature_indexes[0]],
                      ismean=param_i == 0, algo=algo,
                      xlabels=xlabels, ylabels=ylabels, xticks=xticks, yticks=yticks)

    print_heatmap(utils_data.get_robot(), var1="X4", var2="X6", ismean=True, algo="Synthetic")
    print_heatmap(utils_data.get_robot(), var1="X2", var2="X6", ismean=False, algo="Synthetic")
    print_heatmap(get_sin(), var1="X4", var2="X5", ismean=False, algo="Synthetic")

    ebm = stats_all[0]['ebm_inter']
    ebm_dist = stats_all[0]['ebm_dist_inter']
    ebm_dist_unpure = stats_all[0]['ebm_dist_inter_unpure']

    for param_i in range(2):
        for inter_number in range(10):
            print_heatmap_inter(ebm_dist, param_i=param_i, inter_number=inter_number)
            print_heatmap_inter(ebm_dist_unpure, param_i=param_i, inter_number=inter_number, pure=False)

    # print_heatmap(ebm.additive_terms_[6], var1="X4", var2="X6", ismean=True, algo="EBM")
    for inter_number in range(10):
        print_heatmap_inter(ebm, param_i=0, inter_number=6 + inter_number, is_engb=False)
