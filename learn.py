import os
import random
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.neural_network import MLPClassifier
import pandas as pd
import ga

"""
Contains all machine-learning code, which includes cross-validation support, splitting of data into
training, validation, and testing, and feature selection using a genetic algorithm.
"""


NUM_PARTICIPANTS = 6
MAX_ITER = 10000
FINAL_RESULTS_SVM_ITER = 1000
FINAL_RESULTS_MLP_ITER = 100

# Initialize the dictionary that will hold the average scores from each model technique. For now, initialize all the
# scores to 0.
model_scores = {
    'all_data_svm': 0,
    'feature_engineered_svm': 0,
    'all_data_mlp': 0,
    'feature_engineered_mlp': 0
}

top_feature_sets = {
    'feature_engineered_svm': None,
    'feature_engineered_mlp': None
}

# Set the current working directory and the participant data directory.
cwd = os.getcwd()
participant_dir = os.path.join(cwd, 'Data', 'Participants')
participant_ids = [0, 1, 2, 3, 6, 7]

temp_df = []
for id in participant_ids:
    f = os.path.join(participant_dir, f'{str(id)}_Participant_Data.csv')
    temp_df.append(pd.read_csv(f, header=0))

df = pd.concat([temp_df[0], temp_df[1], temp_df[2], temp_df[3], temp_df[4], temp_df[5]], sort=False)
df = df.drop(columns=['Unnamed: 32'])
original_headers = list(df.columns.values)  # All headers in the data. Use this to utilize all features.

# The overall Train-Validation set, used by the GA in feature-selecting the best features and training the model.
# Omits the fifth session data from each player.
df_tv = df.drop(4, axis=0)
X_tv = df_tv.to_numpy()

# The held-out test set, used for final evaluation of the model after feature selection has taken place.
df_test = df.drop([0, 1, 2, 3], axis=0)

# Classes (player IDs) the sessions belong to.
y = np.array(["0", "0", "0", "0", "0",
              "1", "1", "1", "1", "1",
              "2", "2", "2", "2", "2",
              "3", "3", "3", "3", "3",
              "6", "6", "6", "6", "6",
              "7", "7", "7", "7", "7"])

# Classes the sessions belong to, only for the training-validation set.
y_tv = np.delete(y, [4, 9, 14, 19, 24, 29])

# Classes the sessions belong to, only for the test set.
y_test = np.array(["0", "1", "2", "3", "6", "7"])

# Set the groups for LeaveOneGroupOut. The groups are session groups so each player has 4 of 5 sessions in the training
# set and 1 of 5 sessions in the test set.
logo_groups = np.array([1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5])

# Sets the groups for LeaveOneGroupOut, but only for the overall train-validation set, leaving out the fifth session of
# each player as a final model evaluation test set.
logo_groups_2 = np.array([1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4])

"""
Helper functions:
"""


def gen_feature_sets(num_feature_sets, num_original_features, new_pop=None, random_init=False):
    """
    Generates feature sets by translating the chromosome values into the correct feature set.
    """
    current_feature_sets = [{} for i in range(num_feature_sets)]
    for fs in range(num_feature_sets):
        current_features = ['' for l in range(num_original_features)]

        if random_init:
            current_feature_sets[fs] = {
                'id': fs,
                'set': np.array([random.randint(0, 1) for x in range(num_original_features)]),
                'feature_titles': None,
                'fitness': 0.0
            }
        else:
            current_feature_sets[fs] = {
                'id': fs,
                'set': np.array(new_pop[fs]['set']),
                'feature_titles': None,
                'fitness': 0.0
            }

        # Set the named features according to the binary values in the feature set chromosome.
        for f in range(num_original_features):
            if current_feature_sets[fs]['set'][f] == 1:
                current_features[f] = original_headers[f]

        current_feature_sets[fs]['feature_titles'] = [i for i in current_features if i]

    return current_feature_sets


def ga_feature_selection(current_feature_sets, ga_pop, num_original_features, print_to_console=False):
    new_population = []

    for j in range(ga_pop):
        num_features = num_original_features
        weighted_pop = ga.create_weighted_selection(current_feature_sets)
        selected_chromosome_x = ga.selection(weighted_pop)
        selected_chromosome_y = ga.selection(weighted_pop)

        # Produce a child from the two selected x and y chromosomes.
        child = ga.reproduce(selected_chromosome_x, selected_chromosome_y, num_features)

        # Have a chance to mutate the new child feature set.
        if random.randint(0, 10) > 3:
            ga.mutate(child, num_features)

        if print_to_console:
            print(f"\nj# {j}:\nNew child chromosome: {child}")

        new_population.append(child)
    if print_to_console:
        print(f"\n\nNew population of {len(new_population)} chromosomes: "
              f"{new_population}\n******************************\n\n\n")

    return new_population


def all_data_svm(X, y, groups, print_to_console=False):
    """
    Performs LeaveOneGroupOut cross-validation with SVM as the model on all the available data, skipping the
    data engineering step. At the end, returns the average of the accuracy scores obtained.
    Since this is using all the data (even data that is not ideal for differentiating between players),
    this is likely to give a low average score.
    :return: The average prediction accuracy score obtained.
    """

    if print_to_console:
        print("\nBeginning SVM on all data (no feature engineering):\n")

    logo = LeaveOneGroupOut()

    temp_scores = np.zeros(len(groups) // NUM_PARTICIPANTS)
    i = 0
    for train_index, test_index in logo.split(X, y, groups):
        if print_to_console:
            print(f"Training on: {train_index}. Testing on: {test_index}.")
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        clf = SVC(kernel='linear')
        clf.fit(X_train, y_train)  # Train the SVM model.
        temp_scores[i] = clf.score(X_test, y_test)  # Test the model

        if print_to_console:
            print(f"Session group {i + 1} score: {temp_scores[i]}\n")

        i = i + 1

    final_model_score = np.mean(temp_scores)

    if print_to_console:
        print(f"\n\nFinal SVM (without feature engineering) score: {final_model_score}\n\n"
              f"#######################################################################################\n"
              f"#######################################################################################")

    return final_model_score


def engineered_data_svm(y, tv_df, original_headers, groups, ga_iterations=100,
                        ga_pop=320, print_to_console=False):
    """
    Uses a genetic algorithm in combination with SVM models to find (one of) the best feature sets for SVM.
    """
    num_original_features = len(original_headers)

    final_feature_set = {
        'id': None,
        'set': np.ones(32, dtype=np.int),
        'feature_titles': None,
        'fitness': 0.0,
    }

    global_best = 0.0
    new_population = []
    i = 0
    while i < ga_iterations and global_best < 1.0:

        if i == 0:
            current_feature_sets = gen_feature_sets(ga_pop, num_original_features, random_init=True)
        else:
            current_feature_sets = gen_feature_sets(ga_pop, num_original_features,
                                                    new_pop=new_population, random_init=False)

        # Calculate the fitness functions of all feature sets in current_feature_sets
        for c in range(ga_pop):

            # init X based on the current feature sets
            X = tv_df[current_feature_sets[c]['feature_titles']].to_numpy()

            # Test the current feature set to get its predictive accuracy (fitness score).
            current_feature_sets[c]['fitness'] = all_data_svm(X, y, groups, print_to_console=False)

            if print_to_console:
                print(f"Iteration # {i}, feature set # {c} fitness: {current_feature_sets[c]['fitness']}")

            if current_feature_sets[c]['fitness'] > global_best:
                global_best = current_feature_sets[c]['fitness']
                final_feature_set = current_feature_sets[c]
                if print_to_console:
                    print(f"\n\n\n!!!!!!!!!!!!!!!!!!\nNew global best!\n={global_best}\n{final_feature_set}"
                          f"\n\n\n!!!!!!!!!!!!!!!!!!\n\n")

        # Run GA-related functions
        new_population = ga_feature_selection(current_feature_sets, ga_pop, num_original_features, print_to_console)

        i = i + 1

    if print_to_console:
        print(f"\n\nFinal SVM (WITH feature engineering) score: {global_best}\n\n"
              f"With feature set: {final_feature_set['feature_titles']}\n"
              f"#######################################################################################\n"
              f"#######################################################################################")

    return global_best, final_feature_set


def all_data_mlp(X, y, groups, iters=10, print_to_console=False):
    """
    Performs LeaveOneGroupOut cross-validation with MLP as the model on all the available data, skipping the
    data engineering step. At the end, returns the average of the accuracy scores obtained.
    Since this is using all the data (even data that is not ideal for differentiating between players),
    this is likely to give a low average score.
    :return: The average prediction accuracy score obtained.
    """
    if print_to_console:
        print("\nBeginning MLP on all data (no feature engineering):\n")

    logo = LeaveOneGroupOut()

    avg_scores = np.array([0.0 for x in range(iters)])
    i = 0
    while i < iters:

        temp_scores = np.zeros(len(groups) // NUM_PARTICIPANTS)
        j = 0
        for train_index, test_index in logo.split(X, y, groups):
            if print_to_console:
                print(f"Training on: {train_index}. Testing on: {test_index}.")
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            clf = MLPClassifier(solver='lbfgs', hidden_layer_sizes=(20), max_iter=1000)
            clf.fit(X_train, y_train)  # Train the MLP model.
            temp_scores[j] = clf.score(X_test, y_test)  # Test the model

            if print_to_console:
                print(f"Session group {j + 1} score: {temp_scores[j]}\n")

            j = j + 1

        avg_scores[i] = np.mean(temp_scores)
        if print_to_console:
            print(f"\n\nIteration {i} average score: {avg_scores[i]}\n*********\n")

        i = i + 1

    final_model_score = np.mean(avg_scores)

    if print_to_console:
        print(f"\n\nFinal MLP (without feature engineering) score: {final_model_score}\n\n"
              f"#######################################################################################\n"
              f"#######################################################################################")

    return final_model_score


def engineered_data_mlp(y, tf_tv, original_headers, groups, ga_iterations=10,
                        ga_pop=320, print_to_console=False):
    """
    Uses a genetic algorithm in combination with MLP models to find (one of) the best feature sets for MLP.
    """
    num_original_features = len(original_headers)

    if print_to_console:
        print("\nBeginning MLP on all data (WITH feature engineering):\n")

    final_feature_set = {
        'id': None,
        'set': np.ones(32, dtype=np.int),
        'feature_titles': None,
        'fitness': 0.0,
    }

    global_best = 0.0
    new_population = []
    i = 0
    while i < ga_iterations and global_best < 1.0:

        if i == 0:
            current_feature_sets = gen_feature_sets(ga_pop, num_original_features, random_init=True)
        else:
            current_feature_sets = gen_feature_sets(ga_pop, num_original_features,
                                                    new_pop=new_population, random_init=False)

        # Calculate the fitness functions of all feature sets in current_feature_sets
        for c in range(ga_pop):

            # init X
            X = tf_tv[current_feature_sets[c]['feature_titles']].to_numpy()

            # Test the current feature set to get its predictive accuracy (fitness score).
            current_feature_sets[c]['fitness'] = all_data_mlp(X, y, groups, iters=5, print_to_console=False)

            if print_to_console:
                print(f"Iteration # {i}, feature set # {c} fitness: {current_feature_sets[c]['fitness']}")

            if current_feature_sets[c]['fitness'] > global_best:
                global_best = current_feature_sets[c]['fitness']
                final_feature_set = current_feature_sets[c]
                if print_to_console:
                    print(f"\n\n\n!!!!!!!!!!!!!!!!!!\nNew global best!\n={global_best}\n{final_feature_set}"
                          f"\n\n\n!!!!!!!!!!!!!!!!!!\n\n\n")

        # Run GA-related functions
        new_population = ga_feature_selection(current_feature_sets, ga_pop, num_original_features, print_to_console)

        i = i + 1

    if print_to_console:
        print(f"\n\nFinal MLP (WITH feature engineering) score: {global_best}\n\n"
              f"With feature set: {final_feature_set['feature_titles']}\n"
              f"#######################################################################################\n"
              f"#######################################################################################")

    return global_best, final_feature_set


def model_test(df_test, y_test, df_train, y_train, model, feature_set, print_to_console=True):

    print(f"\n\nConducting final model evaluation using feature set: {feature_set}")

    if model == 'SVM':
        X_test = df_test[feature_set].to_numpy()
        X_train = df_train[feature_set].to_numpy()
        clf = SVC(kernel='linear')
        clf.fit(X_train, y_train)                       # Train the SVM model.

    elif model == 'MLP':
        X_test = df_test[feature_set].to_numpy()
        X_train = df_train[feature_set].to_numpy()
        clf = MLPClassifier(solver='lbfgs', hidden_layer_sizes=(50), max_iter=5000)
        clf.fit(X_train, y_train)  # Train the MLP model.

    print(f"Predictions (correct answer is [0, 1, 2, 3, 6, 7]):\n {clf.predict(X_test)}\n\n")

    final_accuracy = clf.score(X_test, y_test)      # Test the model

    if print_to_console:
        print(f"Final Accuracy of chosen model and feature set:\n {final_accuracy}")

    return final_accuracy


"""
Program run starts here:
"""
user_choice = input("Please select an option:\n1: Run SVM with all features.\n2: Run SVM with feature engineering.\n"
                    "3: Run MLP with all features.\n4: Run MLP with feature engineering.\n5: Obtain final accuracy "
                    "results for SVM with best features.\n6: Obtain final accuracy "
                    "results for MLP with best features.")

if user_choice == '1':
    final_accuracy = model_test(df_test, y_test, df_tv, y_tv, 'SVM', original_headers)
elif user_choice == '2':
    model_scores['feature_engineered_svm'], top_feature_sets['feature_engineered_svm'] = \
        engineered_data_svm(y_tv, df_tv, original_headers, logo_groups_2,
                            ga_iterations=100, ga_pop=100, print_to_console=True)
    selected_feature_set = top_feature_sets['feature_engineered_svm']['feature_titles']
    final_accuracy = model_test(df_test, y_test, df_tv, y_tv, 'SVM', selected_feature_set)
elif user_choice == '3':
    final_accuracy = model_test(df_test, y_test, df_tv, y_tv, 'MLP', original_headers)
elif user_choice == '4':
    model_scores['feature_engineered_mlp'], top_feature_sets['feature_engineered_mlp'] = \
        engineered_data_mlp(y_tv, df_tv, original_headers, logo_groups_2, ga_iterations=100,
                            ga_pop=100, print_to_console=True)
    selected_feature_set = top_feature_sets['feature_engineered_mlp']['feature_titles']
    final_accuracy = model_test(df_test, y_test, df_tv, y_tv, 'MLP', selected_feature_set)
elif user_choice == '5':
    intermediate_accuracies = np.zeros(FINAL_RESULTS_SVM_ITER)

    # The ideal feature set found by running the SVM with 1000 GA iterations and 1000 chromosomes.
    ideal_feature_set = ['Session', 'Avg_Resp_Time', 'Low_Start_Y', 'Low_End_Y', 'High_End_Y',
                             'Avg_Between_T_Y', 'Avg_Between_T_X', 'Std_Dev_End_Y', 'Std_Dev_Between_X',
                             'Avg_Velocity', 'Std_Dev_Velocity', 'Non_Rule_Chg_Errs']
    for i in range(FINAL_RESULTS_SVM_ITER):
        intermediate_accuracies[i] = model_test(df_test, y_test, df_tv, y_tv, 'SVM', ideal_feature_set)
    final_accuracy = np.mean(intermediate_accuracies)
    print(f"\n\n\n####################\n####################\n\nFINAL ACCURACY FOR SVM = {final_accuracy}\n\n")
elif user_choice == '6':
    intermediate_accuracies = np.zeros(FINAL_RESULTS_MLP_ITER)

    # The ideal feature set found by running MLP with GA over 100 iterations with 100 chromosomes.
    ideal_feature_set = ['Avg_Resp_Time', 'Low_Resp_Time', 'Low_Start_Y', 'High_Start_Y', 'Avg_Start_Y', 'Avg_End_Y', 'Avg_Between_T_Y',
    'Std_Dev_Start_Y', 'Std_Dev_Between_X', 'Low_Velocity', 'Non_Rule_Chg_Errs', 'Categories_Comp', 'High_Streak',
    'Avg_Streak']
    for i in range(FINAL_RESULTS_MLP_ITER):
        intermediate_accuracies[i] = model_test(df_test, y_test, df_tv, y_tv, 'MLP', ideal_feature_set)
    final_accuracy = np.mean(intermediate_accuracies)
    print(f"\n\n\n####################\n####################\n\nFINAL ACCURACY FOR MLP = {final_accuracy}\n\n")
else:
    print("Invalid selection. Please type '1', '2', '3', '4', '5', or '6'.")
