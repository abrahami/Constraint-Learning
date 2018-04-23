import pandas as pd
import numpy as np
import scipy as sc
import sys
from numpy.random import mtrand
from sklearn.metrics import mean_squared_error
from ast import literal_eval
import matplotlib.pyplot as plt
import seaborn as sns

__author__ = 'abrahami'


class ConstraintRegressor(object):
    """
    class to manage the constrained gradient boosting regressor algorithm.
    It includes functions to be used by the GBT algorithm and variables held inside the object to easily follow the
    algorithm steps and analyse results afterwards

    Parameters
    ----------
    cv_params: dictionary, default: None
        cross validation algorithm parameters. should include 'percentile_threshold' and 'constraint_interval_size'
        in case None is provided the values set are: 'percentile_threshold': 0.1 'constraint_interval_size': 0.05
    gbt_params: dictionary, default: None
        gradient boosting trees parameters. should include 'n_estimators', 'max_depth','min_samples_split',
        'learning_rate' and 'loss'
        in case None is provided the values set are: 'n_estimators': 100, 'max_depth': 5, 'min_samples_split': 1,
        'learning_rate': 0.01, 'loss': 'ls'
    constraints_params: dictionary, default: None
        constraints related parameters. should include 'eta' and 'gamma'
        in case None is provided the values set are: 'eta': 0.1, 'gamma': 1
    constraint_early_stopping: boolean, default: True
        whether or not to apply an early stopping rule to constraints. If set to True, the logic in 'still_candidate'
        function is applied in each step of the algorithm
    dataset: string, default: "boston"
        dataset to be used along the run, must be one out of the following:
        'boston', 'diabetes', 'news', 'ailerons', 'kc_house', 'compactiv_data', 'bikeSharing_data',
        'CTSlices_data', 'Intel_CHT_data', 'parkinson_updrs', 'parkinson_mpower', 'flights', 'Intel_BI_SKL_22'
    test_percent: float, default: 0.3
        percentage of population to be used as a test (i.e. validation) dataset. This % of the population is not
        used at all for learning and building the algorithm but only for validating it
    seed: int, default: 123
        seed value used for building the algorithm. It is useful for running the algorithm twice (or more) in exactly
        the same way

    Attributes
    ----------
    is_constrainted: pandas series (vector of booleans)
        vector with True/False whether each instance is constrainted or not (length: as the # of instances in the data)
    satisfaction_history: dataframe
        dataframe which represents to each instance if its constraint was satisfied or not in the i'th iteration.
        If the instance is not constrained at all, it will have True in all places.
        Number of rows: # of instances in the data
        Number of columns: # of iterations in the algorithm
    constraints_df_train: dataframe
        dataframe containing useful information about the constraints of each instance.
        Columns maintained in this
    """

    def __init__(self, cv_params=None, gbt_params=None, constraints_params=None, constraint_early_stopping=False,
                 dataset="boston", test_percent=0.3, seed=123):

        if cv_params is None:
            cv_params = {'percentile_threshold': 0.9, 'constraint_interval_size': 0.05}
        if gbt_params is None:
            gbt_params = {'n_estimators': 100, 'max_depth': 5, 'min_samples_split': 1, 'learning_rate': 0.01,
                          'loss': 'ls'}
        if constraints_params is None:
            constraints_params = {'eta': 0.1, 'gamma': 1}
        self.percentile_threshold = cv_params['percentile_threshold']
        self.constraint_interval_size = cv_params['constraint_interval_size']
        self.constraint_early_stopping = constraint_early_stopping
        self.dataset = str(dataset)
        self.test_percent = test_percent
        self.seed = seed
        self.n_estimators = gbt_params['n_estimators']
        self.max_depth = gbt_params['max_depth']
        self.min_samples_split = gbt_params['min_samples_split']
        self.learning_rate = gbt_params['learning_rate']
        self.loss = gbt_params['loss']
        self.constraints_eta = constraints_params['eta']
        self.constraints_gamma = constraints_params['gamma']
        self.is_constrainted = None
        self.satisfaction_history = None
        self.constraints_df_train = None

    def still_candidate(self, index, loop_number):
        """
        simple function to return a boolean decision regarding a specific instance - whether it is still 'playing' in
        the algorithm or not. Reason no to 'play' can be:
        1. the instance wasn't constrained at all from the beginning
        2. early stopping criteria - which stoops the efforts of satisfying the constraints on the instance after
        few tries
        constraints of instances w
        :param index: int
            instance index (must be >=0)
        :param loop_number: int
            iteration number in the GBT model (mist be >=1)
        :return: boolean
            True in case the instance is still a candidate to try and satisfy it's constraints, False if not
        """
        # case the early stopping criteria is not turned on or we are at a very early stage of the algorithm
        if (not self.is_constrainted[index]) or (not self.constraint_early_stopping)\
                or(loop_number < self.n_estimators/2.0):
            return True
        # case the early stopping is turned on and the instance doesn't meet the constraints in the last 10 loops
        elif sum(self.satisfaction_history.iloc[index, int(loop_number-self.n_estimators/2.0):loop_number]) == 0:
            return False
        else:
            return True

    def update_satisfaction(self, predictions, loop_number):
        """
        updates the 'satisfaction_history' data-frame with a full column (i.e. for all instances)
        :param predictions: array
            vector with current loop predictions to all instances
        :param loop_number: int
            loop number in the algorithm (must be >=1)
        :return: nothing
            only updates the class object variable('satisfaction_history' variable)
        """
        constraints_df_with_pred = self.constraints_df_train.copy()
        constraints_df_with_pred["pred"] = predictions
        cur_satisfaction = constraints_df_with_pred.apply(lambda x: x['min_value'] <= x["pred"] <= x['max_value'],
                                                          axis=1)
        self.satisfaction_history.iloc[:, loop_number] = cur_satisfaction

    def set_constraints_matrix(self, y, path=None, verbose=False):
        '''
        creates a constraint matrix for a given data. If no path is given, it means that default constraints are
        being generated (default is per dataset, but in most cases it means that part of the data is constrained
        according to the value of the y feature)
        :param y: numpy array
            the target feature, should be a continuous one
        :param path: string
            path to the exact file where the constraint matrix exists. In case it is None, default constraints will be
            generated (depends on the dataset)
        :param verbose: boolean
            whether or not to print a summary of the constraint matrix generated/loaded
        :return: pandas data-frame
            the constraints data-frame. It is a n x 2 dataframe - first column is the minimum value and the second
            column is the maximum value. Non-constrained instances will have values of [-inf, inf] in the matrix.
            Length of the matrix is the same as the size of the y input value

        Examples
        --------
        >>> constraint_obj = ConstraintRegressor()
        >>> constraint_obj.set_constraints_matrix(y=np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]))
        >>> constraint_obj.set_constraints_matrix(y=np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]),
        >>>                                       path=r"C:\Users\abrahami\Documents\Private\Uni\BGU\Thesis\python_code\constraints_matrix_example.csv",
        >>>                                       verbose=True)
        '''

        if path is not None:
            constraints_df = pd.read_csv(path)
            if constraints_df.shape != (len(y), 2):
                raise IOError("shape of the given constraint matrix is not in the right format. It should have two"
                              "columns and the same length as the 'y' feature")
            constraints_df.columns = ['min_value', 'max_value']
            if verbose:
                not_constrainted = np.sum(constraints_df.apply(lambda x: x['min_value'] == float("-inf")
                                                                         and x['max_value'] == float("+inf"), axis=1))
                print "'set_constraints_matrix' function has ended," \
                      " {} instances were constrained out of {}".format(len(y) - not_constrainted, len(y))
            return constraints_df

        # FIRST STEP - set the constraints data-frame with default values of [-inf, inf]
        constraints_df = pd.DataFrame(index=range(0, len(y), 1),
                                      columns=['min_value', 'max_value'],
                                      dtype='float')
        constraints_df['min_value'] = float("-inf")
        constraints_df['max_value'] = float("+inf")

        # SECOND STEP - calculating an indicator vector with the places which are needed to be constrained
        # By default, the most upper values of the y varailbe will be constrained.
        # The 'flights' dataset is special, as we want to restrict the lower values of this data and not the upper ones
        if self.dataset == "flights":#or self.dataset=="kc_house":
            lower_percentile = np.percentile(a=y, q=(1 - self.percentile_threshold) * 100)
            percentile_indic = ((y <= lower_percentile) & (y != 0))

        # the parkinson_mpower dataset is also special, here we want to constraint all the observations with higer vlaue
        # than 0
        elif self.dataset == "parkinson_mpower":
            percentile_indic = (y >= 0)
        # in all other cases, the constraint will be assigned to the instances with the highest values in the dataset
        else:
            upper_percentile = np.percentile(a=y, q=self.percentile_threshold * 100)
            percentile_indic = (y >= upper_percentile)

        # THIRD STEP - generating the max value constraints (note that up to now we have only defined the indicators)
        constraints_df.loc[percentile_indic, 'max_value'] = \
            (y[percentile_indic] + abs(y[percentile_indic] * self.constraint_interval_size))

        # FOURTH STEP - generating the min value constraints
        # parkinson_mpower dataset is special, so we'll generate a specific (min_value) constraint to this one
        if self.dataset == "parkinson_mpower":
            constraints_df.loc[percentile_indic, 'min_value'] = 0
        # these 3 datasets have one sides constraint only, so the min value is the original y value of each observation
        elif self.dataset in ["Intel_CHT_data", "Intel_BI_SKL_22", "kc_house"]:
            constraints_df.loc[percentile_indic, 'min_value'] = y[percentile_indic]
        # all other datasets have 'regular' minimum value
        else:
            constraints_df.loc[percentile_indic, 'min_value'] = \
                (y[percentile_indic] - abs(y[percentile_indic] * self.constraint_interval_size))
        # case verbose is true, we'll print a summary of the procedure
        if verbose:
            not_constrainted = np.sum(constraints_df.apply(lambda x: x['min_value'] == float("-inf")
                                                                          and x['max_value'] == float("+inf"), axis=1))
            print "'set_constraints_matrix' function has ended," \
                  " {} instances were constrained out of {}".format(len(y)-not_constrainted, len(y))

        return constraints_df

    def load_dataset(self, data_path):
        """
        load_data-set function gets a string as input and creates a data-set to work with along the algorithm
        :param data_path: string
            location where the general path of the datasets are stored
        :return: dictionary with 3 values - train data-set, test data-set and the constraints matrix
        """

        # all cases when the dataset name is given explicitly (e.g. "boston"). If it is not given, we will try to find
        # the data in the 'data_path' folder under the name "data.csv" and will take all columns as X and the last as y
        if self.dataset == "boston":
            boston = pd.read_csv(data_path + "\\boston_data\\boston_house_prices.csv")
            X_data = boston.iloc[:, 0:12].copy()
            y_data = boston.iloc[:, 13].copy()
            y_data = np.asarray(y_data)

        elif self.dataset == "diabetes":
            data = pd.read_csv(data_path + "\\diabetes\\diabetes_data.csv")
            X_data = data.iloc[:, 0:8].copy()
            y_data = data.iloc[:, 9].copy()
            y_data = np.asarray(y_data)

        elif self.dataset == "news":
            data = pd.read_csv(data_path + "\\news_data\\OnlineNewsPopularity.csv")
            X_data = data.iloc[:, 1:59].copy()
            y_data = data.iloc[:, 60].copy()
            y_data = np.asarray(y_data)

        elif self.dataset == "ailerons":
            data = pd.read_csv(data_path + "\\ailerons_data\\ailerons.csv")
            X_data = data.iloc[:, 0:39].copy()
            y_data = data.iloc[:, 40].copy()
            y_data = np.asarray(y_data)

        elif self.dataset == "kc_house":
            data = pd.read_csv(data_path + "\\kingCountryHouses_data\\kc_house_data.csv")
            X_data = data.iloc[:, 2:19].copy()
            y_data = data.iloc[:, 20].copy()
            y_data = np.asarray(y_data)

        elif self.dataset == "compactiv_data":
            data = pd.read_csv(data_path + "\\\compactiv_data\\compactiv_data.csv")
            X_data = data.iloc[:, 0:20].copy()
            y_data = data.iloc[:, 21].copy()
            y_data = np.asarray(y_data)

        elif self.dataset == "bikeSharing_data":
            data = pd.read_csv(data_path + "\\bikeSharing_data\\bikeSharing.csv")
            X_data = data.iloc[:, 1:14].copy()
            y_data = data.iloc[:, 15].copy()
            y_data = np.asarray(y_data)

        elif self.dataset == "CTSlices_data":
            data = pd.read_csv(data_path + "\\CTSlices_data\\slice_localization_data.csv")
            X_data = data.iloc[:, 0:384].copy()
            y_data = data.iloc[:, 385].copy()
            y_data = np.asarray(y_data)

        elif self.dataset == "Intel_CHT_data":
            data = pd.read_csv(data_path + "\\CHT_D_step\\CHT_data.csv")
            X_data = data.iloc[:, 0:812].copy()
            y_data = data.iloc[:, 813].copy()
            y_data = np.asarray(y_data)

        elif self.dataset == "parkinson_updrs":
            data = pd.read_csv(data_path + "\\parkinson_updrs\\parkinsons_data.csv")
            X_data = data.iloc[:, 0:20].copy()
            y_data = data.iloc[:, 20].copy()
            y_data = np.asarray(y_data)

        elif self.dataset == "parkinson_mpower":
            data = pd.read_csv(data_path + "\\parkinson_mpower\\mpower_data.csv")
            X_data = data.iloc[:, 1:94].copy()
            y_data = data.iloc[:, 94].copy()
            y_data = np.asarray(y_data)

        elif self.dataset == "flights":
            data = pd.read_csv(data_path + "\\flights\\Boeing-unNorm_Edan.csv")
            X_data = data.iloc[:, 2:37].copy()
            y_data = data.iloc[:, 1].copy()
            y_data = np.asarray(y_data)

        elif self.dataset == "Intel_BI_SKL_22":
            data = pd.read_csv(data_path + "\\BI_SKL_22\\BI_SKL_ready_for_modeling.csv")
            # taking only the first 100 features (business limits)
            X_data = data.iloc[:, 0:100].copy()
            y_data = data.iloc[:, 500].copy()
            y_data = np.asarray(y_data)

        elif self.dataset == "data" or self.dataset == 'nan':
            # case the data.csv file doesn't exist, we will raise an exception
            try:
                data = pd.read_csv(data_path + "\\data.csv")
            except IOError as e:
                print "Problem occurred along loading the data, {}".format(e)
                sys.exit(1)
            data_size = data.shape
            X_data = data.iloc[:, 0:data_size[1]-1].copy()
            y_data = data.iloc[:, -1].copy()
            y_data = np.asarray(y_data)
        else:
            raise IOError("The dataset you specified is not supported in this script. You should provide one out of "
                          "the following datasets: 'boston', 'diabetes', 'news', 'ailerons', 'kc_house', "
                          "'compactiv_data', 'bikeSharing_data', 'CTSlices_data', 'Intel_CHT_data', "
                          "'parkinson_updrs', 'parkinson_mpower', 'flights', 'Intel_BI_SKL_22'. \n"
                          "Another option is to provide your own dataset, it should be named as 'data.csv' and be "
                          "placed in the folder you specified in 'data_path' field under the 'setup_file.json' file")
            sys.exit(1)
        # generating the constraint matrix
        constraints_df = self.set_constraints_matrix(y=y_data)
        # separation the train/test (also the constraint DF)
        rows_cnt = X_data.shape[0]
        # random.seed(constraints_generator_params['seed'])
        rand_object = mtrand.RandomState(seed=self.seed)
        msk = rand_object.rand(rows_cnt) < self.test_percent

        X_train = (X_data[~msk].copy()).reset_index()
        y_train = y_data[~msk].copy()
        constraints_df_train = constraints_df[~msk].copy().reset_index()
        X_test = X_data[msk].copy().reset_index()
        y_test = y_data[msk].copy()
        constraints_df_test = constraints_df[msk].copy().reset_index()

        # updating the constraints object with the most important subject - the constraints
        self.constraints_df_train = constraints_df_train
        self.is_constrainted = ~constraints_df_train.apply(lambda x: x['min_value'] == float("-inf")
                                                                 and x['max_value'] == float("+inf"), axis=1)
        self.satisfaction_history = pd.DataFrame(data=True,
                                                 index=constraints_df_train.index,
                                                 columns=["iter_" + str(i) for i in range(0, self.n_estimators)],
                                                 dtype='bool')
        self.satisfaction_history[self.is_constrainted] = False
        return ({"X_train": X_train,
                 "y_train": y_train,
                 "X_test": X_test,
                 "y_test": y_test,
                 "constraints_df_train": constraints_df_train,
                 "constraints_df_test": constraints_df_test})

    @staticmethod
    def translate_constraints_params(cur_config_dict):
        '''
        assigns values to 'eta' and 'gamma' parameters, according to the configurations got as input.
        These two parameters can be either a simple number (e.g. 2.4) or a list of numbers (e.g. [2, 4, 5.2]).
        Purpose of this function is just to translate the given input into a number or a list to both eta and gamma
        :param cur_config_dict: dictionary
            the input dictionary, where two keys of this dictionary are the 'constraints_eta' and 'constraints_gamma'
            indeed, one (or both) of them can be none, and then the defualt values are assigned
        :return: tuple
            two values tuple - the eta parameter and the gamma parameter. Each of them can be a list of values
        '''
        eta_input = cur_config_dict['constraints_eta']
        gamma_input = cur_config_dict['constraints_gamma']
        eta = None
        gamma = None

        # case eta is a simple number (e.g. 5.0) or nan
        if type(eta_input) is not str:
            # case eta input is nan, it will get the default value (0.1)
            if np.isnan(eta_input):
                eta = 0.1
            elif np.issubdtype(eta_input, np.signedinteger) or np.issubdtype(eta_input, np.floating):
                eta = eta_input
        # case it is a string, should be in the format of a list
        else:
            try:
                eta = literal_eval(eta_input)
            except ValueError as e:
                print "Problem occurred along translating the 'eta' parameter into a value, it should be either a" \
                      "number of a list like (e.g. '[0.1, 0.2]'"
                sys.exit(1)

        # doing the same process for gamma input param
        # case gamma is a simple number (e.g. 5.0)
        if type(gamma_input) is not str:
            # case gamma input is nan, it will get the default value (1.0)
            if np.isnan(gamma_input):
                gamma = 1.0
            elif np.issubdtype(gamma_input, np.signedinteger) or np.issubdtype(gamma_input, np.floating):
                gamma = gamma_input
        # case it is a string
        else:
            try:
                gamma = literal_eval(gamma_input)
            except ValueError as e:
                print "Problem occurred along translating the 'gamma' parameter into a value, it should be either a" \
                      "number of a list like (e.g. '[0.1, 0.2]')"
                sys.exit(1)
        return eta, gamma

    @staticmethod
    def histogram_plot(constraints_df, prediction, saving_path, header=None):
        """
        plotting a histogram of the constrained instances. Values above zero are the ones we satisfied the constraints.
        Value of the graph the the d1 distance from the constraint
        """

        def _distance_calc(x):
            if (x["pred"] >= x['min_value']) & (x["pred"] <= x['max_value']):
                return min(abs(x["pred"] - x['min_value']), abs(x['max_value'] - x["pred"]))
            elif x["pred"] < x['min_value']:
                return -abs(x['min_value'] - x["pred"])
            elif x["pred"] > x['max_value']:
                return -abs(x['pred'] - x["max_value"])
            else:
                print "some error in the _assign_value function! Check it"
                return None

        constraints_df_with_pred = constraints_df.copy()
        constraints_df_with_pred["pred"] = prediction
        # turning off the interactive mode, so plots will not be displayed (they are all saved in a directory)
        # Also cleaning the plt area
        plt.clf()
        plt.ioff()
        # Show the joint distribution using kernel density estimation
        is_constrained = constraints_df_with_pred.apply(lambda x: False if (x['min_value'] == float("-inf")
                                                                  and x['max_value'] == float("inf")) else True, axis=1)
        constraints_df_subset = constraints_df_with_pred.loc[is_constrained]
        total = constraints_df_subset.shape[0]
        distance_from_constraint = constraints_df_subset.apply(_distance_calc, axis=1)
        constraints_satisfied = sum([1 if i > 0 else 0 for i in distance_from_constraint]) * 1.0 / total
        constraints_not_satisfied = 1 - constraints_satisfied
        hist_plot = sns.distplot(distance_from_constraint, color='black')
        for i, rectangle in enumerate(hist_plot.patches):
            cur_x = rectangle.get_x() + rectangle.get_width() * (1 / 2)
            height = rectangle.get_height()
            width = rectangle.get_width()
            # case we are left to the zero value bucket
            if (cur_x + width < 0) and rectangle.get_height() > 0:
                hist_plot.patches[i].set_color('r')
            # case we are right to the zero value bucket
            elif cur_x > 0 and rectangle.get_height() > 0:
                hist_plot.patches[i].set_color('b')
            # case we are on the border of the zero value bucket
            elif rectangle.get_height() > 0:
                hist_plot.patches[i].set_color('violet')

        # adding a text with the proportions of constraints satisfied and those which weren't
        textstr = 'Constraints Satisfied = %.1f%%\nConstraints Not Satisfied=%.1f%%' % \
                  (constraints_satisfied * 100, constraints_not_satisfied * 100)
        # these are matplotlib.patch.Patch properties
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        # place a text box in upper left in axes coords
        hist_plot.text(0.03, 0.97, textstr, transform=hist_plot.transAxes, fontsize=9, verticalalignment='top',
                       bbox=props)
        plt.ylabel('Density', size=16)
        plt.xlabel('Distance', size=16)
        plt.savefig(saving_path)

    @staticmethod
    def scatter_plot(x, y, x_name, y_name, constraints_df, saving_path, header=None):
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
        g = sns.lmplot(x_name, y_name, legend_out=False, data=df, fit_reg=False, hue="is_constrained",
                       scatter_kws={"marker": ".", "s": 6})
        x_min = min(df[x_name])
        x_max = max(df[x_name])
        y_min = min(df[y_name])
        y_max = max(df[y_name])
        tot_min = min(x_min, y_min)
        tot_max = max(x_max, y_max)
        # adding a 45 line
        plt.plot([tot_min, tot_max], [tot_min, tot_max], 'k--', lw=0.5)
        g.set(xlim=(x_min * 0.9, x_max * 1.1))
        g.set(ylim=(y_min * 0.9, y_max * 1.1))

        plt.ylabel(y_name, size=16)
        plt.xlabel(x_name, size=16)
        if header is not None:
            plt.title(header, size=18)
        plt.savefig(saving_path)

    @staticmethod
    def assign_weights(y_true, constraints_df, method="const", const_value=10):
        """
        returns vector of weights, according to given method. Length of the vector is the same as the length of y_true
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
        # constraints_with_true['true'] = y_true
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
            sample_weight[constrainted_idx] = np.abs(normalized_y[constrainted_idx] / margins[constrainted_idx]) * \
                                              (constraints_with_true.shape[0] * 1.0 / sum(is_constrainted))
            # case we got a number less than 1, we'll convert it to 1
            sample_weight[sample_weight < 1] = 1

        return sample_weight

    @staticmethod
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
        n_constraints = float(constraints_df.shape[0] - redundant_constraints)
        mse = mean_squared_error(y_true, y_predicted)
        constraints_with_pred = constraints_df.copy()
        constraints_with_pred['pred'] = y_predicted
        constraints_with_pred['true'] = y_true

        illogical_constraints = np.sum(constraints_with_pred.apply(lambda x: (x['min_value'] > x['true'])
                                                                             or (x['max_value'] < x['true']), axis=1))
        if illogical_constraints:
            print "Error: you have % d illogical constraints in your matrix. Please check all of them and try again." \
                  "" % illogical_constraints
            exit(1)

        # next measure will calculate the # of constraints misses (sum of indicators)
        constraints_misses = np.sum(
            constraints_with_pred.apply(lambda x: not (x['min_value'] <= x['pred'] <= x['max_value']),
                                        axis=1))
        # next measure will calculate the sum of error^2, in case constraint is not satisfied. Only one part of the
        # summation will be relevant, or none of them
        sum_constraints_violation = np.sum(
            constraints_with_pred.apply(lambda x: max(x['min_value'] - x['pred'], 0) ** 2 +
                                                  max(x['pred'] - x['max_value'], 0) ** 2,
                                        axis=1))

        return {'n_constraints': n_constraints, 'MSE': mse,
                'CER': constraints_misses / n_constraints, 'CMSE': sum_constraints_violation / n_constraints}

    def general_weight_loop(self, data, clf, weights):
        X_train = data["X_train"]
        y_train = data["y_train"]
        X_test = data["X_test"]
        y_test = data["y_test"]
        constraints_df_test = data["constraints_df_test"]
        results_df = pd.DataFrame(index=weights,
                                  columns=['n_constraints_train', 'CMSE_train', 'MSE_train', 'CER_train',
                                           'n_constraints_test', 'CMSE_test', 'MSE_test', 'CER_test'],
                                  dtype='float')
        for cur_weight in weights:
            print "\nStarted option B with weight = {}".format(str(cur_weight))
            sample_weight = ConstraintRegressor.assign_weights(y_true=y_train,
                                                               constraints_df=self.constraints_df_train,
                                                               method="const", const_value=cur_weight)
            clf.fit(X_train, y_train, sample_weight=sample_weight, weights_based_constraints_sol=False)
            cur_eval_train = ConstraintRegressor.regression_eval(y_true=y_train, y_predicted=clf.predict(X_train),
                                                                 constraints_df=self.constraints_df_train)
            cur_prediction = clf.predict(X_test)
            cur_eval_test = ConstraintRegressor.regression_eval(y_true=y_test, y_predicted=cur_prediction,
                                                                constraints_df=constraints_df_test)
            results_df.loc[cur_weight, :] = [cur_eval_train['n_constraints'],
                                             cur_eval_train['CMSE'], cur_eval_train['MSE'], cur_eval_train['CER'],
                                             cur_eval_test['n_constraints'],
                                             cur_eval_test['CMSE'], cur_eval_test['MSE'], cur_eval_test['CER']]
            print "Results over the test data set current weight are: " + str(cur_eval_test)
        return results_df

    def dynamic_weight_loop(self, data, clf, etas):
        X_train = data["X_train"]
        y_train = data["y_train"]
        X_test = data["X_test"]
        y_test = data["y_test"]
        constraints_df_test = data["constraints_df_test"]
        results_df = pd.DataFrame(index=etas,
                                  columns=['n_constraints_train', 'CMSE_train', 'MSE_train', 'CER_train',
                                           'n_constraints_test', 'CMSE_test', 'MSE_test', 'CER_test'],
                                  dtype='float')
        for cur_eta in etas:
            print "\nStarted option D with eta = {}".format(str(cur_eta))
            self.constraints_eta = cur_eta
            clf.constraint_obj.constraints_eta = cur_eta
            clf.constraints_eta = cur_eta
            clf.constraints_weights = None
            sample_weight = ConstraintRegressor.assign_weights(y_true=y_train,
                                                               constraints_df=self.constraints_df_train,
                                                               method="const", const_value=10)
            clf.fit(X_train, y_train, sample_weight=sample_weight, weights_based_constraints_sol=True)

            cur_eval_train = ConstraintRegressor.regression_eval(y_true=y_train, y_predicted=clf.predict(X_train),
                                                                 constraints_df=self.constraints_df_train)
            cur_prediction = clf.predict(X_test)
            cur_eval_test = ConstraintRegressor.regression_eval(y_true=y_test, y_predicted=cur_prediction,
                                                                constraints_df=constraints_df_test)
            results_df.loc[cur_eta, :] = [cur_eval_train['n_constraints'],
                                          cur_eval_train['CMSE'], cur_eval_train['MSE'], cur_eval_train['CER'],
                                          cur_eval_test['n_constraints'],
                                          cur_eval_test['CMSE'], cur_eval_test['MSE'], cur_eval_test['CER']]
            print "Results over the test data set current eta are: " + str(cur_eval_test)
        return results_df

    def new_gradinet_loop(self, data, clf, gammas):
        X_train = data["X_train"]
        y_train = data["y_train"]
        X_test = data["X_test"]
        y_test = data["y_test"]
        constraints_df_test = data["constraints_df_test"]
        results_df = pd.DataFrame(index=gammas,
                                  columns=['n_constraints_train', 'CMSE_train', 'MSE_train', 'CER_train',
                                           'n_constraints_test', 'CMSE_test', 'MSE_test', 'CER_test'],
                                  dtype='float')
        for cur_gamma in gammas:
            print "\nStarted option E with gamma = {}".format(str(cur_gamma))
            self.constraints_gamma = cur_gamma
            clf.constraint_obj.constraints_gamma = cur_gamma
            clf.constraints_gamma = cur_gamma
            clf.constraint_obj.loss = "constraints"
            clf.loss = "constraints"
            clf.fit(X_train, y_train, weights_based_constraints_sol=False)
            cur_eval_train = ConstraintRegressor.regression_eval(y_true=y_train, y_predicted=clf.predict(X_train),
                                                                 constraints_df=self.constraints_df_train)
            cur_prediction = clf.predict(X_test)
            cur_eval_test = ConstraintRegressor.regression_eval(y_true=y_test, y_predicted=cur_prediction,
                                                                constraints_df=constraints_df_test)
            results_df.loc[cur_gamma, :] = [cur_eval_train['n_constraints'],
                                            cur_eval_train['CMSE'], cur_eval_train['MSE'], cur_eval_train['CER'],
                                            cur_eval_test['n_constraints'],
                                            cur_eval_test['CMSE'], cur_eval_test['MSE'], cur_eval_test['CER']]
            print "Results over the test data set current gamma are: " + str(cur_eval_test)
        return results_df
