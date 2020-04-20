import csv

import pandas as pd
import numpy as np
import math
import random
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import copy
import sys

# Define constants
CONST_A_ERROR = .00005              # Dataset A epsilon
CONST_B_ERROR = 100                 # Dataset B epsilon
CONST_C_ERROR = 1450                # Dataset C epsilon
CONST_TOTAL_ITERATIONS = 5000       # Iteration limit
FIRED_NEURON_STATE = 1
BLANK_NEURON_STATE = 0


# Calculate and build confusion matrix
def count_confusion(prediction, actual, title):
    true_pos = 0
    true_neg = 0
    false_pos = 0
    false_neg = 0

    for counter in range(0, len(actual)):
        if prediction[counter] == 1 and actual[counter] == 1:
            true_pos = true_pos + 1

    for counter in range(0, len(actual)):
        if prediction[counter] == 0 and actual[counter] == 0:
            true_neg = true_neg + 1

    for counter in range(0, len(actual)):
        if prediction[counter] == 1 and actual[counter] == 0:
            false_pos = false_pos + 1

    for counter in range(0, len(actual)):
        if prediction[counter] == 0 and actual[counter] == 1:
            false_neg = false_neg + 1

    macc = round(true_pos/(true_pos+false_neg)*100, 2)
    facc = round(true_neg/(true_neg+false_pos)*100, 2)

    table = go.Figure(data=[go.Table(name='Data Set A', header=dict(values=[title, 'Predicted F',
                                                                             'Predicted M', 'Total', 'Recognition %']),
                                      cells=dict(values=[['Actual F', 'Actual M', 'Total'],
                                                         [true_pos, false_pos, (true_pos + false_pos)],
                                                         [false_neg, true_neg, (true_neg + false_neg)],
                                                         [(true_pos + false_neg), (true_neg + false_pos), len(prediction)],
                                                         [macc, facc, round((macc + facc) / 2, 2)]]))])

    table.show()
    return


# Helper plot method
def plot(patterns, weights=[0, 0, 0], title="M/F Data"):
    fig, ax = plt.subplots()                            # Define core figure features
    ax.set_title(title)
    ax.set_xlabel("Height")
    ax.set_ylabel("Weight")

    if weights is not None:
        min = 0.0                                       # Set figure scaling
        max = 1.1
        y_res = 0.001
        x_res = 0.001
        y = np.arange(min, max, y_res)
        x = np.arange(min, max, x_res)
        z = []

        for current_y in np.arange(min, max, y_res):    # Assign predictions
            for current_x in np.arange(min, max, x_res):
                z.append(predict([1.0, current_x, current_y], weights))

        x, y = np.meshgrid(x, y)
        z = np.array(z)
        z = z.reshape(x.shape)

    cp = plt.contourf(x, y, z, levels=[-1, -0.0001, 0, 1], colors=('r', 'b'), alpha=0.1)
    class_one_data = [[], []]
    class_two_data = [[], []]

    for i in range(len(patterns)):                      # Separate classes
        current_height = patterns[i][1]
        current_weight = patterns[i][2]
        current_class = patterns[i][-1]

        if current_class == 1:
            class_one_data[0].append(current_height)
            class_one_data[1].append(current_weight)
        else:
            class_two_data[0].append(current_height)
            class_two_data[1].append(current_weight)

    plt.xticks(np.arange(0.0, 1.1, 0.1))                # Plot the data
    plt.yticks(np.arange(0.0, 1.1, 0.1))
    plt.xlim(0, 1.05)
    plt.ylim(0, 1.05)
    class_ones = plt.scatter(class_two_data[0], class_two_data[1], s=40.0, c='b', label='Class Male')
    class_twos = plt.scatter(class_one_data[0], class_one_data[1], s=40.0, c='r', label='Class Female')

    plt.legend(fontsize=10, loc=1)
    plt.show()

    return


# Normalize the dataframe
def normalize_data(df):
    df['Height'] = ((df['Height'] - df['Height'].min()) / (df['Height'].max() - df['Height'].min()))
    df['Weight'] = ((df['Weight'] - df['Weight'].min()) / (df['Weight'].max() - df['Weight'].min()))
    return df


# Convert dataframe entries to floats
def turn_to_float(df):
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    return df


# Add a bias column to the dataframe
def add_bias_col(df):
    bias_column = [1.0] * len(df['Height'])
    df['Bias'] = bias_column
    cols = df.columns.tolist()
    cols = cols[-1:] + cols[:-1]
    df = df[cols]
    return df


# Perform data preparations
def initialize_lists(file):
    df = pd.read_csv(file, names=['Height', 'Weight', 'Class'])
    df = turn_to_float(df)
    df = normalize_data(df)
    df = add_bias_col(df)
    ''' Split data in half'''
    half_length= int(len(df.index)/2)
    df_males = df[:half_length]
    df_females = df[half_length:]

    males_size = len(df_males.index)
    females_size = len(df_females.index)

    ''' Split dataframes into 75% and 25%  with even distribution of males to females '''
    df_75, df_25 =pd.DataFrame(),pd.DataFrame()
    df_75 = df_75.append(df_males[:int(.75*males_size)])
    df_75 = df_75.append(df_females[:int(.75*females_size)])
    df_25 = df_25.append(df_males[int(.75*males_size):])
    df_25 = df_25.append(df_females[int(.75*females_size):])

    '''Convert the dataframes to lists '''
    df_75 = df_75.values.tolist()
    df_25 = df_25.values.tolist()

    '''randomize the indexes '''
    df_75 = sorted(df_75, key = lambda x: random.random() )
    df_25 = sorted(df_25, key = lambda x: random.random() )

    return df_75, df_25


# Initialize weights to random values where -0.5 <= x <= 0.5
def initialize_weights(num_of_weights):
    weights = []
    print("INITIALIZING WEIGHTS")
    for i in range(0, num_of_weights):
        weights.append(random.uniform(-.5, .5))
    print(weights)
    return weights


# Soft activation function
def soft_activation_function(net):
    gain = 0.2
    result = 1 / (1 + (math.exp(-1 * gain * net)))
    return result


# Hard activation function
def hard_activation_function(net):
    if net > 0:
        return 1
    else:
        return 0


# Method used to give generic activation function call
def activation_function(pattern, weights, function):
    net = 0
    for feature, weight in zip(pattern, weights):
        net += feature * weight

    if function == 'hard':
        return hard_activation_function(net)
    else:
        return soft_activation_function(net)


# Returns the predicted neuron state with the given input and weights
def predict(pattern, weights):
    net = 0
    for feature, weight in zip(pattern, weights):
        net += feature * weight

    if net > 0:
        return FIRED_NEURON_STATE
    else:
        return BLANK_NEURON_STATE


# Determines how many are correctly classified as a percentage
def test(test_data, weights):
    correct_count = 0.0
    predictions = []

    for i in range(len(test_data)):
        prediction = predict(test_data[i][:-1], weights)
        predictions.append(prediction)

    return predictions


# Trains the neuron
def train(train_data, weights, alpha, function, const, verbose=False):
    for iteration in range(1, CONST_TOTAL_ITERATIONS+1):

        total_error = 0
        for i in range(0, len(train_data)):
            output = activation_function(train_data[i][:-1], weights, function)
            error = train_data[i][-1] - output
            total_error += math.pow(error, 2)

            for j in range(len(weights)):
                weights[j] = weights[j] + (alpha * error * train_data[i][j])

        if verbose:
            print()
            print("ITERATION: " + str(iteration))
            print("WEIGHTS: " + str(weights))
            print("TOTAL ERROR: " + str(total_error))

        if total_error <= const:
            break

    return weights


''' initialize lists, normalize and in put in float form '''
dfA_75, dfA_25 = initialize_lists("Project_Data/GroupA.txt")
dfB_75, dfB_25 = initialize_lists("Project_Data/GroupB.txt")
dfC_75, dfC_25 = initialize_lists("Project_Data/GroupC.txt")

n = 3
res_dfA_75 = [x[n] for x in dfA_75]
res_dfA_25 = [x[n] for x in dfA_25]
res_dfB_75 = [x[n] for x in dfB_75]
res_dfB_25 = [x[n] for x in dfB_25]
res_dfC_75 = [x[n] for x in dfC_75]
res_dfC_25 = [x[n] for x in dfC_25]

perceptrons = []
test_results = []


# #Converts List to CSV
# csvfile = "Project_Data\GroupA75.csv"
#
# #Assuming res is a flat list
# with open(csvfile, "w") as output:
#     writer = csv.writer(output, lineterminator='\n')
#     for val in dfA_75:
#         writer.writerow([val])
#
# csvfile1 = "Project_Data\GroupA25.csv"
#
# #Assuming res is a flat list
# with open(csvfile1, "w") as output:
#     writer = csv.writer(output, lineterminator='\n')
#     for val in dfA_25:
#         writer.writerow([val])
#
# csvfile2 = "Project_Data\GroupB75.csv"
#
# #Assuming res is a flat list
# with open(csvfile2, "w") as output:
#     writer = csv.writer(output, lineterminator='\n')
#     for val in dfB_75:
#         writer.writerow([val])
#
# csvfile3 = "Project_Data\GroupB25.csv"
#
# #Assuming res is a flat list
# with open(csvfile3, "w") as output:
#     writer = csv.writer(output, lineterminator='\n')
#     for val in dfB_25:
#         writer.writerow([val])
#
# csvfile4 = "Project_Data\GroupC75.csv"
#
# #Assuming res is a flat list
# with open(csvfile4, "w") as output:
#     writer = csv.writer(output, lineterminator='\n')
#     for val in dfC_75:
#         writer.writerow([val])
#
# csvfile5 = "Project_Data\GroupC25.csv"
#
# #Assuming res is a flat list
# with open(csvfile5, "w") as output:
#     writer = csv.writer(output, lineterminator='\n')
#     for val in dfC_25:
#         writer.writerow([val])
#





for i in range(0, 12):
    perceptrons.append(initialize_weights(3))

perceptrons[0] = train(dfA_75, perceptrons[0], 0.3, 'hard', CONST_A_ERROR, True)
perceptrons[1] = train(dfA_75, perceptrons[1], 0.3, 'soft', CONST_A_ERROR, True)
perceptrons[2] = train(dfA_25, perceptrons[2], 0.3, 'hard', CONST_A_ERROR, True)
perceptrons[3] = train(dfA_25, perceptrons[3], 0.3, 'soft', CONST_A_ERROR, True)

perceptrons[4] = train(dfB_75, perceptrons[4], 0.3, 'hard', CONST_B_ERROR, True)
perceptrons[5] = train(dfB_75, perceptrons[5], 0.3, 'soft', CONST_B_ERROR, True)
perceptrons[6] = train(dfB_25, perceptrons[6], 0.3, 'hard', CONST_B_ERROR, True)
perceptrons[7] = train(dfB_25, perceptrons[7], 0.3, 'soft', CONST_B_ERROR, True)

perceptrons[8] = train(dfC_75, perceptrons[8], 0.3, 'hard', CONST_C_ERROR, True)
perceptrons[9] = train(dfC_75, perceptrons[9], 0.3, 'soft', CONST_C_ERROR, True)
perceptrons[10] = train(dfC_25, perceptrons[10], 0.3, 'hard', CONST_C_ERROR, True)
perceptrons[11] = train(dfC_25, perceptrons[11], 0.3, 'soft', CONST_C_ERROR, True)

test_results.append(test(dfA_25, perceptrons[0]))
test_results.append(test(dfA_25, perceptrons[1]))
test_results.append(test(dfA_75, perceptrons[2]))
test_results.append(test(dfA_75, perceptrons[3]))

test_results.append(test(dfB_25, perceptrons[4]))
test_results.append(test(dfB_25, perceptrons[5]))
test_results.append(test(dfB_75, perceptrons[6]))
test_results.append(test(dfB_75, perceptrons[7]))

test_results.append(test(dfC_25, perceptrons[8]))
test_results.append(test(dfC_25, perceptrons[9]))
test_results.append(test(dfC_75, perceptrons[10]))
test_results.append(test(dfC_75, perceptrons[11]))

count_confusion(test_results[0], res_dfA_25, "DATA A - HARD - Test 25%")
plot(dfA_75, perceptrons[0], "DATA A - HARD - Train 75%")
plot(dfA_25, perceptrons[0], "DATA A - HARD - Test 25%")

count_confusion(test_results[1], res_dfA_25, "DATA A - SOFT - Test 25%")
plot(dfA_75, perceptrons[1], "DATA A - SOFT - Train 75%")
plot(dfA_25, perceptrons[1], "DATA A - SOFT - Test 25%")

count_confusion(test_results[2], res_dfA_75, "DATA A - HARD - Test 75%")
plot(dfA_25, perceptrons[2], "DATA A - HARD - Train 25%")
plot(dfA_75, perceptrons[2], "DATA A - HARD - Test 75%")

count_confusion(test_results[3], res_dfA_75, "DATA A - SOFT - Test 75%")
plot(dfA_25, perceptrons[3], "DATA A - SOFT - Train 25%")
plot(dfA_75, perceptrons[3], "DATA A - SOFT - Test 75%")

count_confusion(test_results[4], res_dfB_25, "DATA B - HARD - Test 25%")
plot(dfB_75, perceptrons[4], "DATA B - HARD - Train 75%")
plot(dfB_25, perceptrons[4], "DATA B - HARD - Test 25%")

count_confusion(test_results[5], res_dfB_25, "DATA B - SOFT - Test 25%")
plot(dfB_75, perceptrons[5], "DATA B - SOFT - Train 75%")
plot(dfB_25, perceptrons[5], "DATA B - SOFT - Test 25%")

count_confusion(test_results[6], res_dfB_75, "DATA B - HARD - Test 75%")
plot(dfB_25, perceptrons[6], "DATA B - HARD - Train 25%")
plot(dfB_75, perceptrons[6], "DATA B - HARD - Test 75%")

count_confusion(test_results[7], res_dfB_75, "DATA B - SOFT - Test 75%")
plot(dfB_25, perceptrons[7], "DATA B - SOFT - Train 25%")
plot(dfB_75, perceptrons[7], "DATA B - SOFT - Test 75%")


# print(test_results[8])
# print(res_dfC_25)

count_confusion(test_results[8], res_dfC_25, "DATA C - HARD - Test 25%")
plot(dfC_75, perceptrons[8], "DATA C - HARD - Train 75%")
plot(dfC_25, perceptrons[8], "DATA C - HARD - Test 25%")


count_confusion(test_results[9], res_dfC_25, "DATA C - SOFT - Test 25%")
plot(dfC_75, perceptrons[9], "DATA C - SOFT - Train 75%")
plot(dfC_25, perceptrons[9], "DATA C - SOFT - Test 25%")

# print(test_results[10])
# print(res_dfC_75)

count_confusion(test_results[10], res_dfC_75, "DATA C - HARD - Test 75%")
plot(dfC_25, perceptrons[10], "DATA C - HARD - Train 25%")
plot(dfC_75, perceptrons[10], "DATA C - HARD - Test 75%")

count_confusion(test_results[11], res_dfC_75, "DATA C - SOFT - Test 75%")
plot(dfC_25, perceptrons[11], "DATA C - SOFT - Train 25%")
plot(dfC_75, perceptrons[11], "DATA C - SOFT - Test 75%")
