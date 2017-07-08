#!usr/bin/env python
# Abraham Israeli

import numpy as np
import pandas as pd
import scipy as sc
from sklearn.metrics import mean_squared_error
from sklearn import ensemble_Avrahami_thesis
import matplotlib.pyplot as plt
from numpy.random import mtrand
import seaborn as sns


def regression_eval(y_true, y_predicted, constraints_df):
    """
    regression_eval function gives few evaluation measures to a given solution. List of returned measures is given
    in the 'return' section
    :param y_true: tuple of the true y values, should be a numeric tuple
    :param y_predicted: tuple of the predicted y values, should be a numeric tuple
    :param constraints_df: matrix with the constraints. This matrix is created in the 'load_dataset' and it contains
      2 columns (max/min value) to each observation
    :return: dictionary with all measures:
                n_constraints - # of relevant constraints (those which are not -inf < observation_value < inf)
                MSE = mean squared error
                CER = constraints error rate (% of constraints missed)
                CMSE = constraints mean squared error. Average is over the # of constraints and the sum of error is
                 in regards to to given constraints. Few example:
                 if constraint is 40 <= x <= 60 and  predicted value is 50 - nothing will be added to the sum
                 if constraint is 40 <= x <= 60 and predicted value is 62 - value of 2^2 (4) will be added to the sum
                 if constraint is 40 <= x <= 60 and predicted value is 30 - value of 10^2 (100) will be added to the sum
    """
    # checking if there are redundant constraints
    redundant_constraints = np.sum(constraints_df.apply(lambda x: x['min_value'] == float("-inf")
                                                                  and x['max_value'] == float("+inf"), axis=1))
    n_constraints = float(constraints_df.shape[0]-redundant_constraints)
    mse = mean_squared_error(y_true, y_predicted)
    constraints_with_pred = constraints_df
    constraints_with_pred['pred'] = y_predicted
    constraints_with_pred['true'] = y_true

    illogical_constraints = np.sum(constraints_df.apply(lambda x: (x['min_value'] > x['true'])
                                                                  or (x['max_value'] < x['true']), axis=1))
    if illogical_constraints:
        print "Error: you have % d illogical constraints in your matrix. Please check all of them and try again." \
              "" % illogical_constraints
        exit(1)

    # next measure will calculate the # of constraints misses (sum of indicators)
    constraints_misses = np.sum(constraints_with_pred.apply(lambda x: not(x['min_value'] <= x['pred'] <= x['max_value']),
                                                            axis=1))
    # next measure will calculate the sum of error^2, in case constraint is not satisfied. Only one part of the
    # summation will be relevant, or none of them
    sum_constraints_violation = np.sum(constraints_with_pred.apply(lambda x: max(x['min_value'] - x['pred'], 0)**2 +
                                                                             max(x['pred'] - x['max_value'], 0)**2,
                                                                   axis=1))
    #constraints_with_pred.to_csv(path_or_buf = "C:\\Users\\abrahami\\Documents\\Private\\Uni\\BGU\\Thesis\\Results\\matrix_test.csv")
    return {'n_constraints': n_constraints, 'MSE': mse,
            'CER': constraints_misses/n_constraints, 'CMSE': sum_constraints_violation/n_constraints}


def load_dataset(dataset_name, random_constraints=True, test_percent=0.3, constraints_generator_params=None):
    """
    load_data-set function gets a string as input and creates a data-set to work with along the algorithm
    :param dataset_name: string to work with. Current supported ones are "boston", "diabetes", "news"
    :param random_constraints: whether or not to create random constraints
    :param test_percent: the proportion of data being used for the test population
    :param constraints_generator_params: dictionary of dictionaries containing all parameters to be used when generating
            the constraints. One of the dictionaries is used in case we are creating random constraints (in case we
            don't use a test dataset) and the other dictionary is used in cases where we do split to train/test and then
            the constraints are not fully random
    :return: dictionary with 3 values - train data-set, test data-set and the constraints matrix
    """
    data_path = "C:\\Users\\abrahami\\Documents\\Private\\Uni\\BGU\\Thesis\\data_sets"
    if dataset_name == "boston":
        #boston = datasets.load_boston()
        boston = pd.read_csv(data_path + "\\boston_data\\boston_house_prices.csv")
        X_data = boston.iloc[:, 0:12].copy()
        y_data = boston.iloc[:, 13].copy()
        y_data = np.asarray(y_data)

    if dataset_name == "diabetes":
        data = pd.read_csv(data_path + "\\diabetes\\diabetes_data.csv")
        X_data = data.iloc[:, 0:8].copy()
        y_data = data.iloc[:, 9].copy()
        y_data = np.asarray(y_data)

    if dataset_name == "news":
        data = pd.read_csv(data_path + "\\news_data\\OnlineNewsPopularity.csv")
        X_data = data.iloc[:, 1:59].copy()
        y_data = data.iloc[:, 60].copy()
        y_data = np.asarray(y_data)

    if dataset_name == "ailerons":
        data = pd.read_csv(data_path + "\\ailerons_data\\ailerons.csv")
        X_data = data.iloc[:, 0:39].copy()
        y_data = data.iloc[:, 40].copy()
        y_data = np.asarray(y_data)

    if dataset_name == "kc_house":
        data = pd.read_csv(data_path + "\\kingCountryHouses_data\\kc_house_data.csv")
        X_data = data.iloc[:, 2:19].copy()
        y_data = data.iloc[:, 20].copy()
        y_data = np.asarray(y_data)

    if dataset_name == "compactiv_data":
        data = pd.read_csv(data_path + "\\\compactiv_data\\compactiv_data.csv")
        X_data = data.iloc[:, 0:20].copy()
        y_data = data.iloc[:, 21].copy()
        y_data = np.asarray(y_data)

    if dataset_name == "bikeSharing_data":
        data = pd.read_csv(data_path + "\\bikeSharing_data\\bikeSharing.csv")
        X_data = data.iloc[:, 1:14].copy()
        y_data = data.iloc[:, 15].copy()
        y_data = np.asarray(y_data)

    if dataset_name == "CTSlices_data":
        data = pd.read_csv(data_path + "\\CTSlices_data\\slice_localization_data.csv")
        X_data = data.iloc[:, 0:384].copy()
        y_data = data.iloc[:, 385].copy()
        y_data = np.asarray(y_data)

    if dataset_name == "Intel_CHT_data":
        data = pd.read_csv(data_path + "\\CHT_D_step\\CHT_data.csv")
        X_data = data.iloc[:, 0:812].copy()
        y_data = data.iloc[:, 813].copy()
        y_data = np.asarray(y_data)

    if dataset_name == "parkinson_updrs":
        data = pd.read_csv(data_path + "\\parkinson_updrs\\parkinsons_data.csv")
        X_data = data.iloc[:, 0:20].copy()
        y_data = data.iloc[:, 20].copy()
        y_data = np.asarray(y_data)

    if dataset_name == "parkinson_mpower":
        data = pd.read_csv(data_path + "\\parkinson_mpower\\mpower_data.csv")
        X_data = data.iloc[:, 1:94].copy()
        y_data = data.iloc[:, 94].copy()
        y_data = np.asarray(y_data)

    if dataset_name == "flights":
        data = pd.read_csv(data_path + "\\flights\\Boeing-unNorm_Edan.csv")
        X_data = data.iloc[:, 2:37].copy()
        y_data = data.iloc[:, 1].copy()
        y_data = np.asarray(y_data)

    if dataset_name == "Intel_BI_SKL_22":
        data = pd.read_csv(data_path + "\\BI_SKL_22\\BI_SKL_ready_for_modeling.csv")
        # taking only the first 100 features (business limits)
        X_data = data.iloc[:, 0:100].copy()
        y_data = data.iloc[:, 500].copy()
        y_data = np.asarray(y_data)

    # creating the constraint dataframe (for train + test) right now
    constraints_df = pd.DataFrame(index=range(0, X_data.shape[0], 1),
                                  columns=['min_value', 'max_value'],
                                  dtype='float')
    constraints_df['min_value'] = float("-inf")
    constraints_df['max_value'] = float("+inf")

    # case we want to have fully random constraints (best to do it when we don't have any train/test separation
    # if we are not creating random constraints, we can generate it ourselfs (hand made) - do it your self please...
    if random_constraints and test_percent == 0:
        cur_params = constraints_generator_params['random_params']
        constraints_creator =\
            RandomConstraintsCreator(down_prob_constraint=cur_params['down_prob_constraint'],
                                     up_prob_constraint=cur_params['up_prob_constraint'],
                                     down_mean_constraint=cur_params['down_mean_constraint'],
                                     up_mean_constraint=cur_params['up_mean_constraint'],)
        # case the seed of the creator was defined by the user, if not, it will be created as a random one
        if 'seed' in constraints_generator_params:
            constraints_creator.seed = constraints_generator_params['seed']
        constraints_df = constraints_creator.generate_constraints(y_true=y_data)
        return({"X_train": X_data,
                "y_train": y_data,
                "X_test": None,
                "y_test": None,
                "constraints_df_train": constraints_df,
                "constraints_df_test": None})
    if test_percent > 0:
        cur_params = constraints_generator_params['cv_params']
        # separation the train/test (also the constraint DF)
        rows_cnt = X_data.shape[0]
        #random.seed(constraints_generator_params['seed'])
        rand_object = mtrand.RandomState(seed=constraints_generator_params['seed'])
        msk = rand_object.rand(rows_cnt) < test_percent

        X_train = (X_data[~msk].copy()).reset_index()
        y_train = y_data[~msk].copy()
        constraints_df_train = constraints_df[~msk].copy().reset_index()
        X_test = X_data[msk].copy().reset_index()
        y_test = y_data[msk].copy()
        constraints_df_test = constraints_df[msk].copy().reset_index()

        # assigning values to the constraint df (train and test)
        # taking a special approach with the 'flights' dataset', as we want to restrict the lower values of this data
        if dataset_name == "flights":
            train_percentile_down = np.percentile(a=y_train, q=(1 - cur_params['percentile_threshold']) * 100)
            # train_percentile_up = np.percentile(a=y_train, q=cur_params['percentile_threshold'] * 100)
            # train_percentile_indic = (((y_train <= train_percentile_down) | (y_train >= train_percentile_up))
            #                          & (y_train != 0))
            train_percentile_indic = ((y_train <= train_percentile_down) & (y_train != 0))
            test_percentile_down = np.percentile(a=y_test, q=(1 - cur_params['percentile_threshold']) * 100)
            test_percentile_indic = ((y_test <= test_percentile_down) & (y_test != 0))

        elif dataset_name == "parkinson_mpower":
            train_percentile_indic = (y_train >= 0)
            test_percentile_indic = (y_test >= 0)
        else:
            train_percentile = np.percentile(a=y_train, q=cur_params['percentile_threshold'] * 100)
            train_percentile_indic = (y_train >= train_percentile)
            test_percentile = np.percentile(a=y_test, q=cur_params['percentile_threshold'] * 100)
            test_percentile_indic = (y_test >= test_percentile)

        # Now - we'll generate the constraints themselfs (up to now we have only defined the indicators)
        if dataset_name == "parkinson_mpower":
            constraints_df_train.loc[train_percentile_indic, 'min_value'] = 0
            constraints_df_test.loc[test_percentile_indic, 'min_value'] = 0
            # min value constraint will be generated to all datasets
        else:
            constraints_df_train.loc[train_percentile_indic, 'min_value'] =\
                (y_train[train_percentile_indic]-abs(y_train[train_percentile_indic]*
                                                     cur_params['constraint_interval_size']))
            constraints_df_test.loc[test_percentile_indic, 'min_value'] =\
                    (y_test[test_percentile_indic]-abs(y_test[test_percentile_indic]*
                                                       cur_params['constraint_interval_size']))
            # max value constraint will not be generated to the Intel use cases, because of business reasons
            if dataset_name != "Intel_CHT_data" and dataset_name != "Intel_BI_SKL_22":
                constraints_df_train.loc[train_percentile_indic, 'max_value'] =\
                    (y_train[train_percentile_indic]+
                     abs(y_train[train_percentile_indic]*cur_params['constraint_interval_size']))
                constraints_df_test.loc[test_percentile_indic, 'max_value'] =\
                    (y_test[test_percentile_indic]+
                     abs(y_test[test_percentile_indic]*cur_params['constraint_interval_size']))

        return({"X_train": X_train,
                "y_train": y_train,
                "X_test": X_test,
                "y_test": y_test,
                "constraints_df_train": constraints_df_train,
                "constraints_df_test": constraints_df_test})


def train_constraints_eta(data, constraints_dfs, params, etas=np.arange(0, 0.1, 0.01)):
    """
    train_constraints_eta is a loop over our algorithm, which in each iteration runs the algorithm with differnt eta
    (eta range is defined according to the user input)
    :param data: dictionary with the X_train, X_test, y_train and y_test
    :constraints_dfs: dictionary with the 2 data-frames of the constraints
    :param params: dictionary of parameters to the algorithm. Example can be seen in the main file
    :param etas: range of etas to check
    :return: a data-frame, where each row is a different run (with different eta) and each column is a measure
    """

    X_train = data.get("X_train")
    y_train = data.get("y_train")
    train_constraints_df = constraints_dfs.get("constraints_df_train")
    CER = []
    CMSE = []
    MSE = []

    for i in etas:
        params['constraints_eta'] = i
        clf = ensemble_Avrahami_thesis.GradientBoostingRegressor(**params)
        sample_weight = assign_weights(y_true=y_train, constraints_df=train_constraints_df, method="const",
                                       const_value=1)
        clf.fit(X_train, y_train, sample_weight=sample_weight, weights_based_constraints_sol=True)

        cur_eval = regression_eval(y_true=data.get("y_test"), y_predicted=clf.predict(data.get("X_test")),
                                   constraints_df=constraints_dfs.get("constraints_df_test"))
        CER.append(cur_eval['CER'])
        CMSE.append(cur_eval['CMSE'])
        MSE.append(cur_eval['MSE'])
        print "loop with eta equals {} has ended, current reuslts are: CER: {}, CMSE = {}, MSE={}".\
            format(i, cur_eval['CER'], cur_eval['CMSE'], cur_eval['MSE'])
    eval_table = pd.DataFrame({'eta': etas, 'CER': CER, 'CMSE': CMSE, 'MSE': MSE})
    return eval_table


def assign_weights(y_true, constraints_df, method="const", const_value=10):
    """
    returns vector of weights, accroding to given method. Length of the vector is the same as the length of y_true
    :param y_true:
    :param constraints_df: matrix with the constraints. This matrix is created in the 'load_dataset' and it contains
      2 columns (max/min value) to each observation
    :param method: the methods to be used for setting the weights, can be "const"/"constraints_relative"
    :param const_value: if method is "const", then this value is used as weight to relevant observations
    :return: tuple of weights
    """

    if method not in ("const", "constraints_relative"):
        print "You have to provide a valid method name. Please try again"
        return -1

    if const_value < 1:
        print "Warning: The constant value should better be > 1, as all observations get weight of 1 by default"

    constraints_with_true = constraints_df
    #constraints_with_true['true'] = y_true
    constraints_with_true['true'] = y_true[list(constraints_df.index)]
    # setting values of weights to all the observations to 1, only the constrained ones will be changed
    sample_weight = np.ones(shape=constraints_with_true.shape[0])
    is_constrainted = ~constraints_with_true.apply(lambda x: x['min_value'] == float("-inf")
                                                             and x['max_value'] == float("+inf"), axis=1)
    constrainted_idx = np.where(is_constrainted)
    # logic here is to give high weight to observations with constraint, same value to any observation
    if method == "const":
        sample_weight[constrainted_idx] = const_value

    # logic here is to take the following factors:
    # a. margins from the constraints (inversely)
    # b. distance of the true value from the mean (in std scale) - this is the z transform
    # c. # of constraints compared to the population size (this is given as factor to all constraints)
    if method == "constraints_relative":
        down_margins = constraints_with_true.apply(lambda x: (x['true'] - x['min_value']), axis=1)
        down_margins[np.isinf(down_margins)] = 0
        up_margins = constraints_with_true.apply(lambda x: (x['max_value'] - x['true']), axis=1)
        up_margins[np.isinf(up_margins)] = 0
        margins = np.array(down_margins + up_margins)
        normalized_y = sc.stats.mstats.zscore(y_true)
        sample_weight[constrainted_idx] = np.abs(normalized_y[constrainted_idx]/margins[constrainted_idx]) *\
                                          (constraints_with_true.shape[0]*1.0/sum(is_constrainted))
        # case we got a number less than 1, we'll convert it to 1
        sample_weight[sample_weight < 1] = 1

    return sample_weight


def plot_constraints_models(x, y, x_name, y_name, constraints_df, saving_path, header=None):
    """
    Simple plotting function - predicted VS true values scatter plot

    """
    # turning off the interactive mode, so plots will not be displayed (they are all saved in a directory).
    # Also cleaning the plt area
    plt.ioff()
    plt.clf()
    # Show the joint distribution using kernel density estimation
    is_constrained = constraints_df.apply(lambda x: 0 if (x['min_value'] == float("-inf")
                                                          and x['max_value'] == float("inf")) else 1, axis=1)
    df = pd.DataFrame({x_name: x, y_name: y, 'is_constrained': is_constrained})
    sns.set_style("ticks")
    g = sns.lmplot(x_name, y_name,legend_out=False,data=df,fit_reg=False,hue="is_constrained",
                   scatter_kws={"marker": ".","s": 6})
    x_min = min(df[x_name])
    x_max = max(df[x_name])
    y_min = min(df[y_name])
    y_max = max(df[y_name])
    tot_min = min(x_min, y_min)
    tot_max = max(x_max,y_max)
    # adding a 45 line
    plt.plot([tot_min, tot_max], [tot_min, tot_max], 'k--', lw=0.5)
    g.set(xlim=(x_min * 0.9, x_max * 1.1))
    g.set(ylim=(y_min * 0.9, y_max * 1.1))

    plt.ylabel(y_name, size=16)
    plt.xlabel(x_name, size=16)
    if header is not None:
        plt.title(header, size=18)
    plt.savefig(saving_path)


def plot_constrainted_instances(constraints_df, saving_path, header=None):
    """
    plotting a histogram of the constrained instances. Values above zero are the ones we satisfied the constraints.
    Value of the graph the the d1 distance from the constraint

    """
    def _distance_calc(x):
        if (x["pred"] >= x['min_value']) & (x["pred"] <= x['max_value']):
            return min(abs(x["pred"]-x['min_value']), abs(x['max_value']-x["pred"]))
        elif x["pred"] < x['min_value']:
            return -abs(x['min_value']-x["pred"])
        elif x["pred"] > x['max_value']:
            return -abs(x['pred'] - x["max_value"])
        else:
            print "some error in the _assign_value function! Check it"
            return None
    # turning off the interactive mode, so plots will not be displayed (they are all saved in a directory)
    # Also cleaning the plt area
    plt.clf()
    plt.ioff()
    # Show the joint distribution using kernel density estimation
    is_constrained = constraints_df.apply(lambda x: False if (x['min_value'] == float("-inf")
                                                          and x['max_value'] == float("inf")) else True, axis=1)
    constraints_df_subset = constraints_df.loc[(is_constrained)]
    total = constraints_df_subset.shape[0]
    distance_from_constraint = constraints_df_subset.apply(_distance_calc, axis=1)
    constraints_satisfied = sum([1 if i>0 else 0 for i in distance_from_constraint])*1.0/total
    constraints_not_satisfied = 1-constraints_satisfied
    hist_plot = sns.distplot(distance_from_constraint, color='black')
    for i, rectangle in enumerate(hist_plot.patches):
        cur_x = rectangle.get_x() + rectangle.get_width() * (1 / 2)
        height = rectangle.get_height()
        width = rectangle.get_width()
        # case we are left to the zero value bucket
        if (cur_x + width < 0) and rectangle.get_height() > 0:
            hist_plot.patches[i].set_color('r')
            #hist_plot.text(x=cur_x+width/2, y=height + 0.015, rotation=-45,
            #               s='{0:.2f}'.format(height), ha="center", color='r', fontsize=7)
        # case we are right to the zero value bucket
        elif cur_x > 0 and rectangle.get_height() > 0:
            hist_plot.patches[i].set_color('b')
            #hist_plot.text(x=cur_x+width/2, y=height + 0.015, rotation=-45,
            #               s='{0:.2f}'.format(height), ha="center", color='b', fontsize=7)
        # case we are on the border of the zero value bucket
        elif rectangle.get_height() > 0:
            hist_plot.patches[i].set_color('violet')
            #hist_plot.text(x=cur_x+width/2, y=height + 0.015, rotation=-45,
            #               s='{0:.2f}'.format(height), ha="center", color='violet', fontsize=7)

    # adding a text with the proportions of constraints satisfied and those which weren't
    textstr = 'Constraints Satisfied = %.1f%%\nConstraints Not Satisfied=%.1f%%' %\
              (constraints_satisfied*100, constraints_not_satisfied*100)
    # these are matplotlib.patch.Patch properties
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    # place a text box in upper left in axes coords
    hist_plot.text(0.03, 0.97, textstr, transform=hist_plot.transAxes, fontsize=9, verticalalignment='top', bbox=props)
    plt.ylabel('Density', size=16)
    plt.xlabel('Distance', size=16)
    plt.savefig(saving_path)


def read_configuration(path):
    config_df = pd.read_csv("C:\\Users\\abrahami\\Documents\\Private\\Uni\\BGU\\Thesis\\python_code\\python_config.csv")


########################################################################################################################
class RandomConstraintsCreator():

    def __init__(self, down_prob_constraint, up_prob_constraint, down_mean_constraint, up_mean_constraint,
                 seed=mtrand.randint(2, 1000)):
        self.down_prob_constraint = down_prob_constraint
        self.up_prob_constraint = up_prob_constraint
        self.down_mean_constraint = down_mean_constraint
        self.up_mean_constraint = up_mean_constraint
        self.seed = seed

    def generate_constraints(self, y_true):
        rand_object = mtrand.RandomState(seed=self.seed)
        sample_size = y_true.shape[0]
        constraints_df = pd.DataFrame(index=range(0, sample_size, 1), columns=['min_value', 'max_value'], dtype='float')
        constraints_df['min_value'] = float("-inf")
        constraints_df['max_value'] = float("+inf")
        down_constraints_indic = rand_object.binomial(n=1, p=self.down_prob_constraint, size=sample_size)
        up_constraints_indic = rand_object.binomial(n=1, p=self.up_prob_constraint, size=sample_size)
        # changing the mean by using the inverse of the lof-normal distribution, in order to get 'at the end' the
        # desired mean (otherwise, the mean would be exp(mean)+sigma^2/2)
        down_shift = rand_object.lognormal(mean=float(np.log(self.down_mean_constraint)+1/8), sigma=0.5, size=sample_size)
        up_shift = rand_object.lognormal(mean=float(np.log(self.up_mean_constraint)+1/8), sigma=0.5, size=sample_size)
        for i in range(0, sample_size):
            if down_constraints_indic[i]:
                (constraints_df.iloc[i])['min_value'] = y_true[i] - down_shift[i]
            if up_constraints_indic[i]:
                (constraints_df.iloc[i])['max_value'] = y_true[i] + up_shift[i]

        return constraints_df







