#!usr/bin/env python
# Abraham Israeli
# CER = constraints error rate
# MSE = mean squared error
# CMSE = constraints mean squared error

from sklearn import constraints_ensamble
import pandas as pd
from datetime import datetime
from ast import literal_eval
import csv
import copy
import json
import os.path
from constraint_regressor import ConstraintRegressor

__author__ = 'abrahami'

setup_details = json.load(open("setup_file.json"))
config_file_loc = setup_details['paths']['configuration']
data_loc = setup_details['paths']['data']
results_loc = setup_details['paths']['results']

###############################################################################
# reading configuration from a file with all config needed to be run
config_df = pd.read_csv(config_file_loc)
save_row_level_predictions = literal_eval(setup_details["saving_options"]["save_row_level_predictions"])
save_evaluation_measures = literal_eval(setup_details["saving_options"]["save_evaluation_measures"])
save_plots = literal_eval(setup_details["saving_options"]["save_plots"])
run_benchmarks = literal_eval(setup_details["benchmarks"]["run_benchmarks"])
constant_weight_values = literal_eval(setup_details["benchmarks"]["constant_weight_values"])

if run_benchmarks:
    run_option_a = True
    run_option_b = True
    run_option_c = True
else:
    run_option_a = run_option_b = run_option_c = False
run_option_d = True
run_option_e = True

# Loop over all configurations in the config file (.csv)
for j in range(config_df.shape[0]):
    cur_config = dict(config_df.loc[j])
    eta, gamma = ConstraintRegressor.translate_constraints_params(cur_config)
    print "Current configurations are as follow: \n" + str(cur_config)
    constraint_reg_obj = ConstraintRegressor(cv_params={'percentile_threshold': cur_config['percentile_threshold'],
                                                        'constraint_interval_size': cur_config['constraint_interval_size']},
                                             gbt_params={'n_estimators': cur_config['n_estimators'],
                                                         'max_depth': cur_config['max_depth'],
                                                         'min_samples_split': cur_config['min_samples_split'],
                                                         'learning_rate': cur_config['learning_rate'],
                                                         'loss': cur_config['loss']},
                                             constraints_params={'eta': eta,
                                                                 'gamma': gamma},
                                             constraint_early_stopping=cur_config['constraint_early_stopping'],
                                             dataset=cur_config['dataset'], test_percent=cur_config['test_percent'],
                                             seed=cur_config['gbt_seed'])
    data = constraint_reg_obj.load_dataset(data_path=data_loc)
    X_train = data["X_train"]
    y_train = data["y_train"]
    constraints_df_train = data["constraints_df_train"]
    X_test = data["X_test"]
    y_test = data["y_test"]
    constraints_df_test = data["constraints_df_test"]

    config_details = [str(datetime.now()), cur_config['index'], constraint_reg_obj.dataset,
                      constraint_reg_obj.percentile_threshold, constraint_reg_obj.constraint_interval_size,
                      constraint_reg_obj.n_estimators, constraint_reg_obj.max_depth,
                      constraint_reg_obj.min_samples_split, constraint_reg_obj.learning_rate, constraint_reg_obj.loss,
                      constraint_reg_obj.constraints_eta, constraint_reg_obj.seed, constraint_reg_obj.constraints_gamma,
                      constraint_reg_obj.constraint_early_stopping]

    # defining the output file we'll write to (logs and results)
    if save_evaluation_measures:
        file_exists = os.path.isfile(results_loc + '\\python_auto_results.csv')
        writer = csv.writer(open(results_loc + '\\python_auto_results.csv', "a"), lineterminator='\n', dialect='excel')
        # case the file doesn't exist, we'll create it and put a header to it
        if not file_exists:
            writer.writerow(['time', 'index', 'Dataset', 'percentile_threshold', 'constraint_interval_size',
                              'n_estimators', 'max_depth', 'min_samples_split', 'learning_rate', 'loss',
                              'constraints_eta', 'random_state', 'constraint_gamma', 'early_stopping', 'Algo_type',
                              'train / test', 'Duration(sec)', 'Constraints_count', 'CMSE', 'MSE', 'CER'])
        excel_writer = pd.ExcelWriter(results_loc + '\\looping_results_index_' +
                                      str(cur_config['index']) + '.xlsx')
        # adding a sheet to the excel, with all configurations (this is the meta-data sheet
        df_to_save = pd.DataFrame(config_details, index=['time', 'index', 'Dataset', 'percentile_threshold',
                                                         'constraint_interval_size', 'n_estimators', 'max_depth',
                                                         'min_samples_split', 'learning_rate', 'loss',
                                                         'constraints_eta', 'random_state', 'constraint_gamma',
                                                         'early_stopping']).transpose()
        df_to_save.to_excel(excel_writer, sheet_name='configurations', index=False)
        excel_writer.save
    clf = constraints_ensamble.GradientBoostingRegressor(constraint_obj=constraint_reg_obj)

    # case we want to save the results, we are creating a data-frame which will hold the saved data at the end
    if save_row_level_predictions:
        row_level_df_resuls = constraints_df_train.append(constraints_df_test)
        row_level_df_resuls["dataset"] = 'train'
        row_level_df_resuls.iloc[constraints_df_train.shape[0]:, 3] = 'test'
        row_level_df_resuls["is_constrainted"] = ~row_level_df_resuls.apply(lambda x: x['min_value'] == float("-inf")
                                                                                      and x['max_value'] == float("+inf"), axis=1)
        row_level_df_resuls["y_true"] = list(y_train)+list(y_test)

    # now building a model with the constraints - doing it for each of the options (a to e)
    # Option A - not taking into consideration the restrictions at all
    if run_option_a:
        start_time = datetime.now()
        clf.fit(X_train, y_train, weights_based_constraints_sol=False)
        duration = (datetime.now() - start_time).seconds
        print "Option A - not taking constraints into consideration at all:"
        cur_train_eval = ConstraintRegressor.regression_eval(y_true=y_train, y_predicted=clf.predict(X_train),
                                                             constraints_df=constraint_reg_obj.constraints_df_train)
        print "TRAIN dataset:" + str(cur_train_eval)

        # handling the test data
        cur_prediction = clf.predict(X_test)
        cur_test_eval = ConstraintRegressor.regression_eval(y_true=y_test, y_predicted=cur_prediction,
                                                            constraints_df=constraints_df_test)
        print "TEST dataset:" + str(cur_test_eval)

        # case we want to save evaluation results to a file
        if save_evaluation_measures:
            cur_train_full_log = copy.copy(config_details)
            cur_train_full_log.extend(("Option A - no constraints", "train", duration, cur_train_eval['n_constraints'],
                                       cur_train_eval['CMSE'], cur_train_eval['MSE'], cur_train_eval['CER']))
            writer.writerow(cur_train_full_log)

            cur_test_full_log = copy.copy(config_details)
            cur_test_full_log.extend(("Option A - no constraints", "test", duration, cur_test_eval['n_constraints'],
                                      cur_test_eval['CMSE'], cur_test_eval['MSE'], cur_test_eval['CER']))
            writer.writerow(cur_test_full_log)
        # case we want to save prediction of each observation
        if save_row_level_predictions:
            row_level_df_resuls["standard_gbt"] = list(clf.predict(X_train)) + list(cur_prediction)
        # case we want to save plots of the algorithm - there are 2 plots we are creating to each method (a to e)
        if save_plots:
            ConstraintRegressor.scatter_plot(x=y_test, y=cur_prediction, x_name='True Value', y_name='Predicted Value',
                                             constraints_df=constraints_df_test,
                                             saving_path=results_loc + "\\option_a_config_no_"
                                                         + str(cur_config['index'])+".jpg")
            ConstraintRegressor.histogram_plot(constraints_df=constraints_df_test,
                                               prediction=cur_prediction,
                                               saving_path=results_loc + "\\histogram_option_a_config_no_"
                                                           + str(cur_config['index'])+".jpg")

    # Option B - taking the restrictions in a  general attitude - giving global weights to the constrained instances
    if run_option_b:
        start_time = datetime.now()
        if type(constant_weight_values) is list and len(constant_weight_values) > 1:
            general_weight_loop_results = constraint_reg_obj.general_weight_loop(data=data, clf=clf,
                                                                                 weights=constant_weight_values)
            general_weight_loop_results.to_excel(excel_writer, sheet_name='general_weight_loop')
            excel_writer.save

        # single run without running over few initial weights (this is in case the 'constant_weight_values' param was
        # a list with a single value or just an integer/float (and not a list at all)
        else:
            # we will check if the input is a list, if it is a list, we'll just pull the first element in it
            if type(constant_weight_values) is list:
                weight_value = constant_weight_values[0]
            # case it is just a number, we'll take it as is
            else:
                weight_value = constant_weight_values
            sample_weight = ConstraintRegressor.assign_weights(y_true=y_train,
                                                               constraints_df=constraint_reg_obj.constraints_df_train,
                                                               method="const", const_value=weight_value)
            clf.fit(X_train, y_train, sample_weight=sample_weight, weights_based_constraints_sol=False)
            duration = (datetime.now() - start_time).seconds
            print "\nOption B - giving general weights to the constrained instances:"
            cur_train_eval = ConstraintRegressor.regression_eval(y_true=y_train, y_predicted=clf.predict(X_train),
                                                                 constraints_df=constraint_reg_obj.constraints_df_train)
            print "TRAIN dataset:" + str(cur_train_eval)

            # handling the test data
            cur_prediction = clf.predict(X_test)
            cur_test_eval = ConstraintRegressor.regression_eval(y_true=y_test, y_predicted=cur_prediction,
                                                                constraints_df=constraints_df_test)
            print "TEST dataset:" + str(cur_test_eval)

            # case we want to save evaluation results to a file
            if save_evaluation_measures:
                cur_train_full_log = copy.copy(config_details)
                cur_train_full_log.extend(("Option B - global weight to constrainted", "train", duration,
                                           cur_train_eval['n_constraints'], cur_train_eval['CMSE'],
                                           cur_train_eval['MSE'], cur_train_eval['CER']))
                writer.writerow(cur_train_full_log)
                cur_test_full_log = copy.copy(config_details)
                cur_test_full_log.extend(("Option B - global weight to constrainted", "test", duration,
                                          cur_test_eval['n_constraints'], cur_test_eval['CMSE'], cur_test_eval['MSE'],
                                          cur_test_eval['CER']))
                writer.writerow(cur_test_full_log)

            # case we want to save prediction of each observation
            if save_row_level_predictions:
                row_level_df_resuls["constant_weight"] = list(clf.predict(X_train)) + list(cur_prediction)
            # case we want to save plots of the algorithm - there are 2 plots we are creating to each method (a to e)
            if save_plots:
                ConstraintRegressor.scatter_plot(x=y_test, y=cur_prediction, x_name='True Value', y_name='Predicted Value',
                                                 constraints_df=constraints_df_test,
                                                 saving_path=results_loc + "\\option_b_config_no_"
                                                                    + str(cur_config['index'])+".jpg")
                ConstraintRegressor.histogram_plot(constraints_df=constraints_df_test,
                                                   prediction=cur_prediction,
                                                   saving_path=results_loc + "\\histogram_option_b_config_no_"
                                                               + str(cur_config['index'])+".jpg")

    # Option C - taking the restrictions into account by setting weights according to the constraint's difficultness
    if run_option_c:
        if constraint_reg_obj.constraint_interval_size == 0:
            print "constraint_interval_size parameter is equal to zero, so option C (weight according to the " \
                  "constraint's difficultness will not run - it is not relevant to use it in such case"
        else:
            start_time = datetime.now()
            sample_weight = ConstraintRegressor.assign_weights(y_true=y_train,
                                                               constraints_df=constraint_reg_obj.constraints_df_train,
                                                               method="constraints_relative")
            clf.fit(X_train, y_train, sample_weight=sample_weight, weights_based_constraints_sol=False)
            duration = (datetime.now() - start_time).seconds
            print "\nOption C - giving \"personal\" weight to observations, according to constraint's difficultness:"
            cur_train_eval = ConstraintRegressor.regression_eval(y_true=y_train, y_predicted=clf.predict(X_train),
                                                                 constraints_df=constraint_reg_obj.constraints_df_train)
            print "TRAIN dataset:" + str(cur_train_eval)

            cur_prediction = clf.predict(X_test)
            cur_test_eval = ConstraintRegressor.regression_eval(y_true=y_test, y_predicted=cur_prediction,
                                                                constraints_df=constraints_df_test)
            print "TEST dataset:" + str(cur_test_eval)

            # case we want to save evaluation results to a file
            if save_evaluation_measures:
                cur_train_full_log = copy.copy(config_details)
                cur_train_full_log.extend(("Option C - smart weight to constrainted", "train", duration,
                                           cur_train_eval['n_constraints'], cur_train_eval['CMSE'],
                                           cur_train_eval['MSE'], cur_train_eval['CER']))
                writer.writerow(cur_train_full_log)

                cur_test_full_log = copy.copy(config_details)
                cur_test_full_log.extend(("Option C - smart weight to constrainted", "test", duration,
                                          cur_test_eval['n_constraints'], cur_test_eval['CMSE'], cur_test_eval['MSE'],
                                          cur_test_eval['CER']))
                writer.writerow(cur_test_full_log)
            # case we want to save prediction of each observation
            if save_row_level_predictions:
                row_level_df_resuls["smart_weight"] = list(clf.predict(X_train)) + list(cur_prediction)
            # case we want to save plots of the algorithm - there are 2 plots we are creating to each method (a to e)
            if save_plots:
                ConstraintRegressor.scatter_plot(x=y_test, y=cur_prediction, x_name='True Value',
                                                 y_name='Predicted Value', constraints_df=constraints_df_test,
                                                 saving_path=results_loc + "\\option_c_config_no_"
                                                             + str(cur_config['index']) + ".jpg")
                ConstraintRegressor.histogram_plot(constraints_df=constraints_df_test,
                                                   prediction=cur_prediction,
                                                   saving_path=results_loc + "\\histogram_option_c_config_no_"
                                                               + str(cur_config['index']) + ".jpg")

    # Option D - taking the constraints per each loop and learning according to the 1st approach
    # (weights change per loop)
    if run_option_d:
        start_time = datetime.now()
        clf = constraints_ensamble.GradientBoostingRegressor(constraint_obj=constraint_reg_obj)
        if type(constraint_reg_obj.constraints_eta) is list:
            etas = constraint_reg_obj.constraints_eta
            dynamic_weight_loop_results = constraint_reg_obj.dynamic_weight_loop(data=data, clf=clf, etas=etas)
            dynamic_weight_loop_results.to_excel(excel_writer, sheet_name='dynamic_weight_loop')
            excel_writer.save
        # single run without running over few etas
        else:
            sample_weight = ConstraintRegressor.assign_weights(y_true=y_train,
                                                               constraints_df=constraint_reg_obj.constraints_df_train,
                                                               method="const", const_value=1)
            clf.fit(X_train, y_train, sample_weight=sample_weight, weights_based_constraints_sol=True)
            duration = (datetime.now() - start_time).seconds
            print "\nOption D - learning according to our 1st model approach - weights change after each loop:"
            cur_train_eval = ConstraintRegressor.regression_eval(y_true=y_train, y_predicted=clf.predict(X_train),
                                                                 constraints_df=constraint_reg_obj.constraints_df_train)
            print "TRAIN dataset:" + str(cur_train_eval)

            cur_prediction = clf.predict(X_test)
            cur_test_eval = ConstraintRegressor.regression_eval(y_true=y_test, y_predicted=cur_prediction, constraints_df=constraints_df_test)

            print "TEST dataset:" + str(cur_test_eval)

            # case we want to save evaluation results to a file
            if save_evaluation_measures:
                cur_train_full_log = copy.copy(config_details)
                cur_train_full_log.extend(("Option D - our 1st algo - dynamic weight", "train", duration,
                                           cur_train_eval['n_constraints'], cur_train_eval['CMSE'],
                                           cur_train_eval['MSE'], cur_train_eval['CER']))
                writer.writerow(cur_train_full_log)

                cur_test_full_log = copy.copy(config_details)
                cur_test_full_log.extend(("Option D - our 1st algo - dynamic weight", "test", duration,
                                          cur_test_eval['n_constraints'], cur_test_eval['CMSE'], cur_test_eval['MSE'],
                                          cur_test_eval['CER']))
                writer.writerow(cur_test_full_log)
            # case we want to save prediction of each observation
            if save_row_level_predictions:
                row_level_df_resuls["dynamic_weight"] = list(clf.predict(X_train))+ list(cur_prediction)
            # case we want to save plots of the algorithm - there are 2 plots we are creating to each method (a to e)
            if save_plots:
                ConstraintRegressor.scatter_plot(x=y_test, y=cur_prediction, x_name='True Value', y_name='Predicted Value',
                                                 constraints_df=constraints_df_test,
                                                 saving_path=results_loc +"\\option_d_config_no_"
                                                             + str(cur_config['index'])+".jpg")
                ConstraintRegressor.histogram_plot(constraints_df=constraints_df_test,
                                                   prediction=cur_prediction,
                                                   saving_path=results_loc + "\\histogram_option_d_config_no_"
                                                               + str(cur_config['index'])+".jpg")

    # Option E - taking the constraints per each loop and learning according to 2nd approach (loss+gradient change)
    if run_option_e:
        start_time = datetime.now()
        constraint_reg_obj.loss = "constraints"
        clf = constraints_ensamble.GradientBoostingRegressor(constraint_obj=constraint_reg_obj)

        if type(constraint_reg_obj.constraints_gamma) is list:
            gammas = constraint_reg_obj.constraints_gamma
            new_gradient_loop_results = constraint_reg_obj.new_gradinet_loop(data=data, clf=clf, gammas=gammas)
            new_gradient_loop_results.to_excel(excel_writer, sheet_name='new_gradient_loop')
            excel_writer.save
        else:
            clf.fit(X_train, y_train, weights_based_constraints_sol=False)
            duration = (datetime.now() - start_time).seconds
            print "\nOption E - learning according to our 2nd model approach - loss and gradient change:"
            cur_train_eval = ConstraintRegressor.regression_eval(y_true=y_train, y_predicted=clf.predict(X_train),
                                                                 constraints_df=constraint_reg_obj.constraints_df_train)
            print "TRAIN dataset:" + str(cur_train_eval)

            cur_prediction = clf.predict(X_test)
            cur_test_eval = ConstraintRegressor.regression_eval(y_true=y_test, y_predicted=cur_prediction,
                                                                constraints_df=constraints_df_test)
            print "TEST dataset:" + str(cur_test_eval)

            # case we want to save evaluation results to a file
            if save_evaluation_measures:
                cur_train_full_log = copy.copy(config_details)
                cur_train_full_log.extend(("Option E - our 2nd algo - new loss function", "train", duration,
                                           cur_train_eval['n_constraints'], cur_train_eval['CMSE'],
                                           cur_train_eval['MSE'],
                                           cur_train_eval['CER']))
                writer.writerow(cur_train_full_log)
                cur_test_full_log = copy.copy(config_details)
                cur_test_full_log.extend(("Option E - our 2nd algo - new loss function", "test", duration,
                                          cur_test_eval['n_constraints'], cur_test_eval['CMSE'], cur_test_eval['MSE'],
                                          cur_test_eval['CER']))
                writer.writerow(cur_test_full_log)
            # case we want to save prediction of each observation
            if save_row_level_predictions:
                row_level_df_resuls["constrint_loss"] = list(clf.predict(X_train)) + list(cur_prediction)
            # case we want to save plots of the algorithm - there are 2 plots we are creating to each method (a to e)
            if save_plots:
                ConstraintRegressor.scatter_plot(x=y_test, y=cur_prediction, x_name='True Value',
                                                 y_name='Predicted Value',
                                                 constraints_df=constraints_df_test,
                                                 saving_path=results_loc + "\\option_e_config_no_"
                                                             + str(cur_config['index']) + ".jpg")
                ConstraintRegressor.histogram_plot(constraints_df=constraints_df_test,
                                                   prediction=cur_prediction,
                                                   saving_path=results_loc + "\\histogram_option_e_config_no_"
                                                               + str(cur_config['index']) + ".jpg")

    # end of big loop
    if save_row_level_predictions:
        row_level_df_resuls.to_csv(path_or_buf=results_loc + "\\row_level_predictions_config_no_" +
                                               str(cur_config['index'])+".csv")
# delete the csv writer, as we have just finished a loop over all options
if save_evaluation_measures:
    del writer
    del excel_writer
