import numpy as np
import csv
from sys import argv
import os
import fnmatch
import math

"""
Takes per-session CSV files and outputs one summary CSV file every time it's run (one for each
participant).
This is a CLI program with the only argument being the participant's ID number that we want 
to summarize.
"""

participant_id = int(argv[1])
ROUND_DIGITS = 3  # Number of digits for all float data to round to after the decimal point.
NUM_TRIALS = 48

# Set the current working directory, the sessions data directory, and the participant data directory.
cwd = os.getcwd()
session_dir = os.path.join(cwd, 'Data', 'Sessions')
participant_dir = os.path.join(cwd, 'Data', 'Participants')
if not os.path.exists(participant_dir):
    os.mkdir(participant_dir)

# Participant data file name stem = absolute path + participant ID.
data_file = os.path.join(participant_dir, f'{participant_id}_Participant_Data.csv')


def calc_session_time(data):
    """
    Sums up all the values in the RespTime column to get several response time associated metrics from the given
    session data.
    :param data: data in 2D numpy array format
    :return: the total session time (time taken by participant to complete test), average response time,
    lowest response time, highest response time, standard deviation of response times, and the average of response times
    of those trials just after rule changes.
    """
    session_time = 0.0
    lowest_time = math.inf
    highest_time = 0.0
    resp_times = np.array([])
    post_rule_chg_resp_times = np.array([])
    i = 0
    category_completed = False
    for row in data:
        resp_times = np.append(resp_times, float(row.item(3)))
        session_time += resp_times.item(i)
        if resp_times[i] > highest_time:
            highest_time = resp_times[i]
        if resp_times[i] < lowest_time:
            lowest_time = resp_times[i]

        # If a category was completed the previous trial, consider the response time to be a post rule change
        # response time and reset the category_completed flag.
        if category_completed:
            post_rule_chg_resp_times = np.append(post_rule_chg_resp_times, float(resp_times.item(i)))
            category_completed = False

        # If the a category was completed this trial, flag the next trial as a post-rule-change trial.
        if row.item(15) == '1':
            category_completed = True
        i += 1

    avg_resp_time = session_time / NUM_TRIALS
    avg_post_rule_chg_resp_time = float(np.mean(post_rule_chg_resp_times, dtype=np.float64))
    std_dev_resp_time = float(np.std(resp_times))
    return round(session_time, ROUND_DIGITS), round(avg_resp_time, ROUND_DIGITS), round(lowest_time, ROUND_DIGITS), \
           round(highest_time, ROUND_DIGITS), round(std_dev_resp_time, ROUND_DIGITS), \
           round(avg_post_rule_chg_resp_time, ROUND_DIGITS)


def calc_session_positions(data):
    """
    Calculates session data related to mouse positions and returns all of them. Most of the x-axis position data is not
    calculated because that data is based mostly on the randomness of the cards and not the behavior of the participant.
    :param data: data in 2D numpy array format
    :return: The lowest starting y position, highest starting position, lowest ending y position, highest ending y
    position, average starting y position, average ending y position, average between-trial y movement
    (measure of previous trial's end y position minus the current trial's starting y position),
    average between-trial x movement(measure of previous trial's end x position minus the current trial's
    starting x position), standard deviation of starting y positions, standard deviation of ending y positions,
    standard deviation of between-trial y movements, and standard deviation of between-trial x movements.
    """
    lowest_start_y = math.inf
    highest_start_y = 0.0
    lowest_end_y = math.inf
    highest_end_y = 0.0
    start_ys = np.array([])
    end_ys = np.array([])
    start_xs = np.array([])
    end_xs = np.array([])
    bt_y_movements = np.array([])
    bt_x_movements = np.array([])
    i = 0
    for row in data:
        start_ys = np.append(start_ys, float(row.item(5)))
        end_ys = np.append(end_ys, float(row.item(7)))
        start_xs = np.append(start_xs, float(row.item(4)))
        end_xs = np.append(end_xs, float(row.item(6)))
        if lowest_start_y > start_ys[i]:  # Check for lowest starting y pos.
            lowest_start_y = start_ys[i]
        if highest_start_y < start_ys[i]:  # Check for highest starting y pos.
            highest_start_y = start_ys[i]
        if lowest_end_y > end_ys[i]:  # Check for lowest ending y pos.
            lowest_end_y = end_ys[i]
        if highest_end_y < end_ys[i]:  # Check for highest ending y pos.
            highest_end_y = end_ys[i]

        if i != 0:
            bt_y_movements = np.append(bt_y_movements, float(end_ys[i - 1] - start_ys[i]))
            bt_x_movements = np.append(bt_x_movements, float(end_xs[i - 1] - start_xs[i]))
        i += 1

    avg_start_y = float(np.mean(start_ys, dtype=np.float64))
    avg_end_y = float(np.mean(end_ys, dtype=np.float64))
    avg_bt_y_movement = float(np.mean(bt_y_movements, dtype=np.float64))
    avg_bt_x_movement = float(np.mean(bt_x_movements, dtype=np.float64))
    std_dev_start_y = float(np.std(start_ys))
    std_dev_end_y = float(np.std(end_ys))
    std_dev_bt_y_movement = float(np.std(bt_y_movements))
    std_dev_bt_x_movement = float(np.std(bt_x_movements))

    return round(lowest_start_y, ROUND_DIGITS), round(highest_start_y, ROUND_DIGITS), \
           round(lowest_end_y, ROUND_DIGITS), round(highest_end_y, ROUND_DIGITS), \
           round(avg_start_y, ROUND_DIGITS), round(avg_end_y, ROUND_DIGITS), \
           round(avg_bt_y_movement, ROUND_DIGITS), round(avg_bt_x_movement, ROUND_DIGITS), \
           round(std_dev_start_y, ROUND_DIGITS), round(std_dev_end_y, ROUND_DIGITS), \
           round(std_dev_bt_y_movement, ROUND_DIGITS), round(std_dev_bt_x_movement, ROUND_DIGITS)


def calc_session_velocity(data):
    """
    Calculates and returns session data related to velocity.
    :param data: data in 2D numpy array format
    :return: Lowest mouse velocity, highest mouse velocity, average mouse velocity
    and standard deviation of mouse velocities.
    """
    lowest_veloc = math.inf
    highest_veloc = 0.0
    mouse_velocities = np.array([])
    i = 0
    for row in data:
        mouse_velocities = np.append(mouse_velocities, float(row.item(8)))
        if lowest_veloc > mouse_velocities[i]:  # Check for lowest mouse velocity.
            lowest_veloc = mouse_velocities[i]
        if highest_veloc < mouse_velocities[i]:  # Check for highest mouse velocity.
            highest_veloc = mouse_velocities[i]
        i += 1

    avg_veloc = float(np.mean(mouse_velocities, dtype=np.float64))
    std_dev_veloc = float(np.std(mouse_velocities))

    return round(lowest_veloc, ROUND_DIGITS), round(highest_veloc, ROUND_DIGITS), \
           round(avg_veloc, ROUND_DIGITS), round(std_dev_veloc, ROUND_DIGITS)


def calc_session_misclicks(data):
    """
    Calculates and returns the total number of misclicks made by the participant for the given session.
    :param data: data in 2D numpy array format
    :return: the total number of misclicks for the given session
    """
    num_misclicks = 0
    for row in data:
        num_misclicks += int(row.item(9))

    return num_misclicks


def calc_session_errors(data):
    """
    Calculates and returns session data related to participant errors.
    :param data: data in 2D numpy array format
    :return: Non-perseverative errors, perseverative errors, and non-rule-change errors (those errors made when there
    was not just a rule change).
    """
    non_pers_errs = 0
    pers_errs = 0
    non_rule_chg_errs = 0
    rule_changed = True

    for row in data:
        error_occurred = False
        if row.item(10) == '1':
            non_pers_errs += 1
            error_occurred = True
        elif row.item(10) == '2':
            pers_errs += 1
            error_occurred = True

        if error_occurred and rule_changed == False:
            non_rule_chg_errs += 1
            rule_changed = False

        if row.item(15) == 1:
            rule_changed = True

    return round(non_pers_errs, ROUND_DIGITS), round(pers_errs, ROUND_DIGITS), round(non_rule_chg_errs, ROUND_DIGITS)


def calc_categories_completed(data):
    """
    Calculates and returns session data related to category completetion and selection correctness.
    :param data: data in 2D numpy array format
    :return: Number of categories completed by participant, highest streak of correct selections, and average streak
    of correct selections.
    """
    highest_streak = 0
    categories_completed = 0
    streaks = np.array([])

    for row in data:
        streaks = np.append(streaks, int(row.item(11)))
        if row.item(15) == '1':
            categories_completed += 1
        if int(row.item(11)) > highest_streak:
            highest_streak = int(row.item(11))

    avg_streak = float(np.mean(streaks, dtype=np.float64))

    return categories_completed, highest_streak, round(avg_streak, ROUND_DIGITS)


def calc_rule_stats(data):
    """
    Calculates and returns session data related to which rules the participant did best with and worst with. Since
    rules are randomly selected, this is done by calculating the fraction of rule correctness over the total number of
    times the rule appeared in the session.
    :param data: data in 2D numpy array format
    :return: Rule type most commonly correctly selected by participant, and rule type least commonly correctly
    selected by participant.
    """
    rule_0_correct = 0
    rule_0_total = 0
    rule_1_correct = 0
    rule_1_total = 0
    rule_2_correct = 0
    rule_2_total = 0

    for row in data:
        current_rule = row.item(2)
        if row.item(10) == '0':
            correct = True
        else:
            correct = False

        if current_rule == '0':
            rule_0_total += 1
            if correct:
                rule_0_correct += 1
        elif current_rule == '1':
            rule_1_total += 1
            if correct:
                rule_1_correct += 1
        elif current_rule == '2':
            rule_2_total += 1
            if correct:
                rule_2_correct +=1

    rule_fractions = np.array([])
    if rule_0_total > 0:
        rule_fractions = np.append(rule_fractions, round(rule_0_correct / rule_0_total, ROUND_DIGITS))
    if rule_1_total > 0:
        rule_fractions = np.append(rule_fractions, round(rule_1_correct / rule_1_total, ROUND_DIGITS))
    if rule_2_total > 0:
        rule_fractions = np.append(rule_fractions, round(rule_2_correct / rule_2_total, ROUND_DIGITS))

    # Get the highest and lowest rule numbers from the indexes of the max and min values inside rule_fractions.
    best_rule = np.where(rule_fractions == np.amax(rule_fractions))[0][0]
    worst_rule = np.where(rule_fractions == np.amin(rule_fractions))[0][0]

    return best_rule, worst_rule


all_sessions_data = []

session_num = 1

for filename in os.listdir(session_dir):
    if fnmatch.fnmatch(filename, str(participant_id) + '*'):
        print(f"Engineering features from {filename}....")
        file = open(os.path.join(session_dir, filename))
        file_reader = csv.reader(file)
        list_data = list(file_reader)
        list_data.pop(0)  # Remove the column headers from the data.
        array_data = np.array(list_data)

        # Calculate time-related statistics.
        session_time, avg_resp_time, lowest_resp_time, highest_resp_time, \
        std_dev_resp_times, avg_post_rule_chg_resp_time = calc_session_time(array_data)

        # Calculate mouse position related statistics.
        lowest_start_y, highest_start_y, lowest_end_y, highest_end_y, \
        avg_start_y, avg_end_y, avg_bt_y_move, avg_bt_x_move, \
        std_dev_start_y, std_dev_end_y, std_dev_bt_y, std_dev_bt_x = calc_session_positions(array_data)

        # Calculate mouse velocity related statistics.
        lowest_veloc, highest_veloc, avg_veloc, std_dev_veloc = calc_session_velocity(array_data)

        # Calculate the number of misclicks for this session.
        misclicks = calc_session_misclicks(array_data)

        # Calculate error-related statistics for this session.
        non_pers_errors, pers_errors, non_rule_chg_errors = calc_session_errors(array_data)

        # Calculate category completed related statistics for this session.
        cats_comp, highest_streak, avg_streak = calc_categories_completed(array_data)

        # Calculate rule-related statistics for this session.
        best_rule, worst_rule = calc_rule_stats(array_data)

        # Create an array of data for this session.
        session_data = {
            "Participant ID": participant_id,
            "Session": session_num,
            "Session time": session_time,
            "Avg response time": avg_resp_time,
            "Lowest response time": lowest_resp_time,
            "Highest response time": highest_resp_time,
            "Std dev response times": std_dev_resp_times,
            "Avg post-rule-change response time": avg_post_rule_chg_resp_time,
            "Lowest start y pos": lowest_start_y,
            "Highest start y pos": highest_start_y,
            "Lowest end y pos": lowest_end_y,
            "Highest end y pos": highest_end_y,
            "Avg start y pos": avg_start_y,
            "Avg end y pos": avg_end_y,
            "Avg between-trial y movement": avg_bt_y_move,
            "Avg between-trial x movement": avg_bt_x_move,
            "Std dev start y pos": std_dev_start_y,
            "Std dev end y pos": std_dev_end_y,
            "Std dev between-trial y movement": std_dev_bt_y,
            "Std dev between-trial x movement": std_dev_bt_x,
            "Lowest mouse velocity": lowest_veloc,
            "Highest mouse velocity": highest_veloc,
            "Average mouse velocity": avg_veloc,
            "Std dev of mouse velocity": std_dev_veloc,
            "Num misclicks": misclicks,
            "Num non-perseverative errors": non_pers_errors,
            "Num perseverative errors": pers_errors,
            "Num non-rule-change errors": non_rule_chg_errors,
            "Num categories completed": cats_comp,
            "Highest streak": highest_streak,
            "Average streak": avg_streak,
            "Most correct rule": best_rule,
            "Least correct rule": worst_rule
        }
        print(f"\n\nSession Data:\n{session_data}\n\n\n\n****************************")
        all_sessions_data.append(session_data)
        session_num += 1
        file.close()


print("\nBeginning to write to Participant File....")

# Open the output file where we'll combine all the session data and write the column names.
output_file = open(os.path.join(participant_dir, data_file), 'w')
output_file.write('Session,Time,Avg_Resp_Time,Low_Resp_Time,High_Resp_Time,Std_Dev_Resp_Time,Avg_Rule_Chg_Resp_Time,')
output_file.write('Low_Start_Y,High_Start_Y,Low_End_Y,High_End_Y,Avg_Start_Y,Avg_End_Y,Avg_Between_T_Y,')
output_file.write('Avg_Between_T_X,Std_Dev_Start_Y,Std_Dev_End_Y,Std_Dev_Between_Y,Std_Dev_Between_X,')
output_file.write('Low_Velocity,High_Velocity,Avg_Velocity,Std_Dev_Velocity,Misclicks,Non_Pers_Errs,Pers_Errs,')
output_file.write('Non_Rule_Chg_Errs,Categories_Comp,High_Streak,Avg_Streak,Best_Rule,Worst_Rule,')
for session in all_sessions_data:
    output_file.write('\n')                         # Go to the next line
    output_file.write(f'{session["Session"]}, {session["Session time"]}, {session["Avg response time"]},')
    output_file.write(f'{session["Lowest response time"]}, {session["Highest response time"]},')
    output_file.write(f'{session["Std dev response times"]}, {session["Avg post-rule-change response time"]},')
    output_file.write(f'{session["Lowest start y pos"]}, {session["Highest start y pos"]},')
    output_file.write(f'{session["Lowest end y pos"]}, {session["Highest end y pos"]},')
    output_file.write(f'{session["Avg start y pos"]}, {session["Avg end y pos"]},')
    output_file.write(f'{session["Avg between-trial y movement"]}, {session["Avg between-trial x movement"]},')
    output_file.write(f'{session["Std dev start y pos"]}, {session["Std dev end y pos"]},')
    output_file.write(f'{session["Std dev between-trial y movement"]}, {session["Std dev between-trial x movement"]},')
    output_file.write(f'{session["Lowest mouse velocity"]}, {session["Highest mouse velocity"]},')
    output_file.write(f'{session["Average mouse velocity"]}, {session["Std dev of mouse velocity"]},')
    output_file.write(f'{session["Num misclicks"]}, {session["Num non-perseverative errors"]},')
    output_file.write(f'{session["Num perseverative errors"]}, {session["Num non-rule-change errors"]},')
    output_file.write(f'{session["Num categories completed"]}, {session["Highest streak"]},')
    output_file.write(f'{session["Average streak"]}, {session["Most correct rule"]}, {session["Least correct rule"]},')

print("\n\nFinished writing to Participant File!")