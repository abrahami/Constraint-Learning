#!usr/bin/env python
# Abraham Israeli
# CER = constraints error rate
# MSE = mean squared error
# CMSE = constraints mean squared error

from sklearn import ensemble_Avrahami_thesis
import pandas as pd
from datetime import datetime
import support_functions as sf
import csv
import copy

general_path = "C:\\Users\\abrahami\\Documents\\Private\\Uni\\BGU\\Thesis"
###############################################################################
# reading configuration from a file with all config needed to be run
config_df = pd.read_csv(general_path + "\\python_code\\python_config.csv")

# Loop over all configurations in the config file we uploaded
for j in range(config_df.shape[0]):
    cur_config = config_df.loc[j]
    print "Current configurations are as follow: \n" + str(cur_config)
    constraints_generator_params = {'seed': cur_config['constraint_seed'],
                                    'random_params': {'down_prob_constraint': cur_config['down_prob_constraint'],
                                                      'up_prob_constraint': cur_config['up_prob_constraint'],
                                                      'down_mean_constraint': cur_config['down_mean_constraint'],
                                                      'up_mean_constraint': cur_config['up_mean_constraint']},
                                    'cv_params': {'percentile_threshold': cur_config['percentile_threshold'],
                                                  'constraint_interval_size': cur_config['constraint_interval_size']}
                                    }
    # loop over same configuration few time (in order to see that our results are stable and not only luck)
    for i in range(1):
        data = sf.load_dataset(dataset_name=cur_config['dataset'],
                               random_constraints=False,
                               test_percent=cur_config['test_percent'],
                               constraints_generator_params=constraints_generator_params)
        X_train = data["X_train"]
        y_train = data["y_train"]
        constraints_df_train = data["constraints_df_train"]
        X_test = data["X_test"]
        y_test = data["y_test"]
        constraints_df_test = data["constraints_df_test"]

        params = {'n_estimators': cur_config['n_estimators'], 'max_depth': cur_config['max_depth'],
                  'min_samples_split': cur_config['min_samples_split'], 'learning_rate': cur_config['learning_rate'],
                  'loss': cur_config['loss'], 'constraints_eta': cur_config['constraints_eta'],
                  'constraints_gamma': cur_config['constraints_gamma'], 'constraints_df': constraints_df_train,
                  'random_state': cur_config['gbt_seed']}
                  # 'SVR_in_leaves': False, 'SVR_parmas': {'kernel': 'sigmoid', 'C': 0.001, 'epsilon': 0.2 , 'degree':3}}

        config_details = [str(datetime.now()),
                          cur_config['index'],
                          cur_config['dataset'],
                          constraints_generator_params['cv_params']['percentile_threshold'],
                          constraints_generator_params['cv_params']['constraint_interval_size'],
                          params['n_estimators'],
                          params['max_depth'],
                          params['min_samples_split'],
                          params['learning_rate'],
                          params['loss'],
                          params['constraints_eta'],
                          params['random_state'],
                          params['constraints_gamma']]

        # defining the output file we'll write to (logs and results)
        writer = csv.writer(open(general_path + "\\Results\\python_auto_results.csv", "a"),
                            lineterminator='\n', dialect='excel')

        # now building a model with the restrictions
        # Option A - not taking into consideration the restrictions at all

        start_time = datetime.now()
        clf = ensemble_Avrahami_thesis.GradientBoostingRegressor(**params)
        clf.fit(X_train, y_train, weights_based_constraints_sol=False)
        duration = (datetime.now() - start_time).seconds
        print "Option A - not taking constraints into consideration at all:"
        cur_eval = sf.regression_eval(y_true=y_train, y_predicted=clf.predict(X_train),
                                      constraints_df=constraints_df_train)
        print "TRAIN dataset:" + str(cur_eval)
        cur_full_log = copy.copy(config_details)
        cur_full_log.extend(("Option A - no constraints",
                             "train",
                             duration,
                             cur_eval['n_constraints'],
                             cur_eval['CMSE'],
                             cur_eval['MSE'],
                             cur_eval['CER']))
        writer.writerow(cur_full_log)

        cur_prediction = clf.predict(X_test)
        cur_eval = sf.regression_eval(y_true=y_test, y_predicted=cur_prediction,
                                      constraints_df=constraints_df_test)
        sf.plot_constraints_models(x=y_test, y=cur_prediction, x_name='True Value', y_name='Predicted Value',
                                   constraints_df=constraints_df_test,
                                   saving_path=general_path + "\\Results\\option_a_config_no_"
                                                            + str(cur_config['index'])+".jpg")
        sf.plot_constrainted_instances(constraints_df=constraints_df_test,
                                       saving_path=general_path +
                                                   "\\Results\\histogram_option_a_config_no_"
                                                   + str(cur_config['index'])+".jpg")

        print "TEST dataset:" + str(cur_eval)
        cur_full_log = copy.copy(config_details)
        cur_full_log.extend(("Option A - no constraints",
                             "test",
                             duration,
                             cur_eval['n_constraints'],
                             cur_eval['CMSE'],
                             cur_eval['MSE'],
                             cur_eval['CER']))
        writer.writerow(cur_full_log)

        # saving the predictions to a file, for future usage - Export this to a function in the future!!
        constraints_df_for_saving = constraints_df_train.append(constraints_df_test)
        constraints_df_for_saving["dataset"] = 'train'
        constraints_df_for_saving.iloc[constraints_df_train.shape[0]:, 5] = 'test'
        constraints_df_for_saving.to_csv(path_or_buf=general_path + "\\Results\\prediction_optionA.csv")

        # Option B - taking the restrictions in a  general attitude - giving global weights to the constrained instances
        start_time = datetime.now()
        sample_weight = sf.assign_weights(y_true=y_train, constraints_df=constraints_df_train, method="const",
                                          const_value=10)
        clf.fit(X_train, y_train, sample_weight=sample_weight, weights_based_constraints_sol=False)
        duration = (datetime.now() - start_time).seconds
        print "\nOption B - giving general weights to the constrained instances:"
        cur_eval = sf.regression_eval(y_true=y_train, y_predicted=clf.predict(X_train),
                                      constraints_df=constraints_df_train)
        print "TRAIN dataset:" + str(cur_eval)
        cur_full_log = copy.copy(config_details)
        cur_full_log.extend(("Option B - global weight to constrainted",
                             "train",
                             duration,
                             cur_eval['n_constraints'],
                             cur_eval['CMSE'],
                             cur_eval['MSE'],
                             cur_eval['CER']))
        writer.writerow(cur_full_log)

        cur_prediction = clf.predict(X_test)
        cur_eval = sf.regression_eval(y_true=y_test, y_predicted=cur_prediction, constraints_df=constraints_df_test)
        sf.plot_constraints_models(x=y_test, y=cur_prediction, x_name='True Value', y_name='Predicted Value',
                                   constraints_df=constraints_df_test,
                                   saving_path=general_path + "\\Results\\option_b_config_no_"
                                                            + str(cur_config['index'])+".jpg")
        sf.plot_constrainted_instances(constraints_df=constraints_df_test,
                                       saving_path=general_path +
                                                   "\\Results\\histogram_option_b_config_no_"
                                                   + str(cur_config['index'])+".jpg")
        print "TEST dataset:" + str(cur_eval)
        cur_full_log = copy.copy(config_details)
        cur_full_log.extend(("Option B - global weight to constrainted",
                             "test",
                             duration,
                             cur_eval['n_constraints'],
                             cur_eval['CMSE'],
                             cur_eval['MSE'],
                             cur_eval['CER']))
        writer.writerow(cur_full_log)

        # saving the predictions to a file, for future usage - Export this to a function in the future!!
        constraints_df_for_saving = constraints_df_train.append(constraints_df_test)
        constraints_df_for_saving["dataset"] = 'train'
        constraints_df_for_saving.iloc[constraints_df_train.shape[0]:, 5] = 'test'
        constraints_df_for_saving.to_csv(path_or_buf=general_path + "\\Results\\prediction_optionB.csv")

        # Option C - taking the restrictions into account by setting weights according to the constraint's difficultness
        start_time = datetime.now()
        sample_weight = sf.assign_weights(y_true=y_train, constraints_df=constraints_df_train, method="constraints_relative")
        clf.fit(X_train, y_train, sample_weight=sample_weight, weights_based_constraints_sol=False)
        duration = (datetime.now() - start_time).seconds
        print "\nOption C - giving \"personal\" weight to observations, according to constraint's difficultness:"
        cur_eval = sf.regression_eval(y_true=y_train, y_predicted=clf.predict(X_train), constraints_df=constraints_df_train)
        print "TRAIN dataset:" + str(cur_eval)
        cur_full_log = copy.copy(config_details)
        cur_full_log.extend(("Option C - smart weight to constrainted",
                             "train",
                             duration,
                             cur_eval['n_constraints'],
                             cur_eval['CMSE'],
                             cur_eval['MSE'],
                             cur_eval['CER']))
        writer.writerow(cur_full_log)

        cur_prediction = clf.predict(X_test)
        cur_eval = sf.regression_eval(y_true=y_test, y_predicted=cur_prediction, constraints_df=constraints_df_test)
        sf.plot_constraints_models(x=y_test, y=cur_prediction, x_name='True Value', y_name='Predicted Value',
                                   constraints_df=constraints_df_test,
                                   saving_path=general_path + "\\Results\\option_c_config_no_"
                                                            + str(cur_config['index'])+".jpg")
        sf.plot_constrainted_instances(constraints_df=constraints_df_test,
                                       saving_path=general_path +
                                                   "\\Results\\histogram_option_c_config_no_"
                                                   + str(cur_config['index'])+".jpg")
        print "TEST dataset:" + str(cur_eval)
        cur_full_log = copy.copy(config_details)
        cur_full_log.extend(("Option C - smart weight to constrainted",
                             "test",
                             duration,
                             cur_eval['n_constraints'],
                             cur_eval['CMSE'],
                             cur_eval['MSE'],
                             cur_eval['CER']))
        writer.writerow(cur_full_log)


        # saving the predictions to a file, for future usage - Export this to a function in the future!!
        constraints_df_for_saving = constraints_df_train.append(constraints_df_test)
        constraints_df_for_saving["dataset"] = 'train'
        constraints_df_for_saving.iloc[constraints_df_train.shape[0]:, 5] = 'test'
        constraints_df_for_saving.to_csv(path_or_buf=general_path + "\\Results\\prediction_optionC.csv")

        # Option D - taking the constraints per each loop and learning according to 1st approach (weights change per loop)
        start_time = datetime.now()
        clf = ensemble_Avrahami_thesis.GradientBoostingRegressor(**params)
        sample_weight = sf.assign_weights(y_true=y_train, constraints_df=constraints_df_train, method="const", const_value=1)
        clf.fit(X_train, y_train, sample_weight=sample_weight, weights_based_constraints_sol=True)
        duration = (datetime.now() - start_time).seconds
        print "\nOption D - learning according to our 1st model approach - weights change after each loop:"
        cur_eval = sf.regression_eval(y_true=y_train, y_predicted=clf.predict(X_train), constraints_df=constraints_df_train)
        print "TRAIN dataset:" + str(cur_eval)
        cur_full_log = copy.copy(config_details)
        cur_full_log.extend(("Option D - our 1st algo - dynamic weight",
                             "train",
                             duration,
                             cur_eval['n_constraints'],
                             cur_eval['CMSE'],
                             cur_eval['MSE'],
                             cur_eval['CER']))
        writer.writerow(cur_full_log)

        cur_prediction = clf.predict(X_test)
        cur_eval = sf.regression_eval(y_true=y_test, y_predicted=cur_prediction, constraints_df=constraints_df_test)
        sf.plot_constrainted_instances(constraints_df=constraints_df_test,
                                       saving_path=general_path +
                                                   "\\Results\\histogram_option_d_config_no_"
                                                   + str(cur_config['index'])+".jpg")
        sf.plot_constraints_models(x=y_test, y=cur_prediction, x_name='True Value', y_name='Predicted Value',
                                   constraints_df=constraints_df_test, #header='Scatter Plot - Train data',
                                   saving_path=general_path +"\\Results\\option_d_config_no_"
                                                            + str(cur_config['index'])+".jpg")
        print "TEST dataset:" + str(cur_eval)
        cur_full_log = copy.copy(config_details)
        cur_full_log.extend(("Option D - our 1st algo - dynamic weight",
                             "test",
                             duration,
                             cur_eval['n_constraints'],
                             cur_eval['CMSE'],
                             cur_eval['MSE'],
                             cur_eval['CER']))
        writer.writerow(cur_full_log)

        # saving the predictions to a file, for future usage - Export this to a function in the future!!
        constraints_df_for_saving = constraints_df_train.append(constraints_df_test)
        constraints_df_for_saving["dataset"] = 'train'
        constraints_df_for_saving.iloc[constraints_df_train.shape[0]:, 5] = 'test'
        constraints_df_for_saving.to_csv(path_or_buf=general_path + "\\Results\\prediction_optionD.csv")

        # Option E - taking the constraints per each loop and learning according to 2nd approach (loss+gradient change)
        start_time = datetime.now()
        params['loss'] = "constraints"
        clf = ensemble_Avrahami_thesis.GradientBoostingRegressor(**params)
        clf.fit(X_train, y_train, weights_based_constraints_sol=False)
        duration = (datetime.now() - start_time).seconds
        print "\nOption E - learning according to our 2nd model approach - loss and gradient change:"
        cur_eval = sf.regression_eval(y_true=y_train, y_predicted=clf.predict(X_train), constraints_df=constraints_df_train)
        print "TRAIN dataset:" + str(cur_eval)
        cur_full_log = copy.copy(config_details)
        cur_full_log.extend(("Option E - our 2nd algo - new loss function",
                             "train",
                             duration,
                             cur_eval['n_constraints'],
                             cur_eval['CMSE'],
                             cur_eval['MSE'],
                             cur_eval['CER']))
        writer.writerow(cur_full_log)

        cur_prediction = clf.predict(X_test)
        cur_eval = sf.regression_eval(y_true=y_test, y_predicted=cur_prediction, constraints_df=constraints_df_test)
        sf.plot_constraints_models(x=y_test, y=cur_prediction, x_name='True Value', y_name='Predicted Value',
                                   constraints_df=constraints_df_test,
                                   saving_path=general_path +"\\Results\\option_e_config_no_"
                                                            + str(cur_config['index'])+".jpg")
        sf.plot_constrainted_instances(constraints_df=constraints_df_test,
                                       saving_path=general_path +
                                                   "\\Results\\histogram_option_e_config_no_"
                                                   + str(cur_config['index'])+".jpg")
        print "TEST dataset:" + str(cur_eval)
        cur_full_log = copy.copy(config_details)
        cur_full_log.extend(("Option E - our 2nd algo - new loss function",
                             "test",
                             duration,
                             cur_eval['n_constraints'],
                             cur_eval['CMSE'],
                             cur_eval['MSE'],
                             cur_eval['CER']))
        writer.writerow(cur_full_log)

        # saving the predictions to a file, for future usage - Export this to a function in the future!!
        constraints_df_for_saving = constraints_df_train.append(constraints_df_test)
        constraints_df_for_saving["dataset"] = 'train'
        constraints_df_for_saving.iloc[constraints_df_train.shape[0]:, 5] = 'test'
        constraints_df_for_saving.to_csv(path_or_buf=general_path + "\\Results\\prediction_optionE.csv")

# delete the csv writer, as we have just finished a loop over all options
del writer
