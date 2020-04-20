import numpy as np
import random
import matplotlib.pyplot as plt

CONST_ITERATIONS = 25000
CONST_ERROR = 0.08
CONST_VERBOSE = True


# Plot the data and regression line
def plot(data, weight_set, title, filename):

    x = []
    y = []
    z = []
    fig = plt.figure()
    
    fig.suptitle(title)
    for j in range(0, len(data)):
        x.append(data[j][1])
        y.append(data[j][-1])
        z.append(activation_function(data[j], weight_set))

    plt.xlabel('Time', fontsize=14)
    plt.ylabel('Consumption', fontsize=14)
    plt.scatter(x, y)
    plt.plot(x, z)
    fig.savefig(filename + ".png")
    plt.show()
    return


# Normalize the dataframe
def normalize_data(x):

    if CONST_VERBOSE:
        print()
        print("DATA BEFORE NORMALIZATION: ")
        print(x)

    x_normed = x / x.max(axis=0)

    if CONST_VERBOSE:
        print()
        print("DATA AFTER NORMALIZATION: ")
        print(x_normed)

    return x_normed


# Perform data preparations
def initialize(file):
    data = np.genfromtxt(file, delimiter=',')
    data.astype(float)
    data = np.column_stack(((np.ones(len(data))), data))

    if CONST_VERBOSE:
        print()
        print("DATA FROM " + file + ": ")
        print(data)

    return data


# Method used to give generic activation function call
def activation_function(x, weight_set):
    net = 0

    for j in range(0, len(weight_set)):
        net += weight_set[j] * x[j]

    return net


def generate_weight_set(num_of_weights):
    weight_set = []

    for x in range(0, num_of_weights):
        weight_set.append(random.uniform(-.5, .5))

    if CONST_VERBOSE:
        print()
        print("INITIALIZING WEIGHTS: " + str(weight_set))

    return weight_set


def train(data, weight_set):
    total_error = 10000000
    count = 1
    if(len(weight_set) >= 4):
        alpha = .3
    else:
        alpha = .01
    while total_error > CONST_ERROR and count < CONST_ITERATIONS:

        if CONST_VERBOSE:
            print()
            print("ITERATION: " + str(count))
            print("INITIAL WEIGHTS: " + str(weight_set))

        total_error = 0
        for x in range(0, len(data)):
            output = activation_function(data[x], weight_set)
            iter_error = data[x][-1] - output
            total_error += iter_error ** 2

            for y in range(0, len(weight_set)):
                weight_set[y] = weight_set[y] + (alpha * iter_error * data[x][y])

        if CONST_VERBOSE:
            print("UPDATED WEIGHTS: " + str(weight_set))
            print("TOTAL ERROR: " + str(total_error))

        count += 1

    return weight_set, total_error


def test(data, weight_set):
    squared_error = 0
    predictions = []

    for x in range(0, len(data)):
        prediction = 0
        for i in range(0, len(weight_set)):
            prediction += data[x][i] * weight_set[i]
        #prediction = activation_function(data, weight_set)
        predictions.append(prediction)
        squared_error += (data[x][-1] - prediction) ** 2

        if CONST_VERBOSE:
            print()
            print("TEST ITERATION: " + str(x))
            print("PREDICTED: " + str(prediction))
            print("ACTUAL: " + str(data[x][1]))

    if CONST_VERBOSE:
        print()
        print("--- TESTING END ---")
        print("SUMMED SQUARED ERROR FOR TEST: " + str(squared_error))
        print("PREDICTED VALUES: " + str(predictions))

    return squared_error, predictions


data_day1 = initialize("Project3_data/train_data_1.txt")
data_day2 = initialize("Project3_data/train_data_2.txt")
data_day3 = initialize("Project3_data/train_data_3.txt")
test_data = initialize("Project3_data/test_data_4.txt")

col = test_data [:,1]
col = np.square(col)
col_order = [0, 1, 3, 2]
test_data = np.column_stack((test_data, col))
test_data = test_data[:,col_order]
col = np.sqrt(col)
col = np.power(col, 3)
col_order = [0, 1, 2, 4, 3]
test_data = np.column_stack((test_data, col))
test_data = test_data[:,col_order]

training_data = np.concatenate((data_day1, data_day2, data_day3), axis=0)
training_data = sorted(training_data, key=lambda x: x[-2])
training_data = [arr.tolist() for arr in training_data]
training_data = np.array(training_data)
training_data_linear = training_data

col = training_data_linear[:,1]
col = np.square(col)
quad_col_order = [0, 1, 3, 2]
training_data_quadratic = np.column_stack((training_data_linear, col))
training_data_quadratic = training_data_quadratic[:,quad_col_order]

col = np.sqrt(col)
col = np.power(col, 3)
cubic_col_order = [0, 1, 2, 4, 3]
training_data_cubic = np.column_stack((training_data_quadratic, col))
training_data_cubic = training_data_cubic[:,cubic_col_order]

test_data = normalize_data(test_data)
training_data_linear = normalize_data(training_data_linear)
training_data_quadratic = normalize_data(training_data_quadratic)
training_data_cubic = normalize_data(training_data_cubic)
ensemble = []
if(CONST_VERBOSE):
    print()
    print("PROCESSED TRAINING DATA SETS")
    print("- - - - - - - - - - - - - - -")
    print()
    print("LINEAR:")
    print(training_data_linear)
    print()
    print("QUADRATIC:")
    print(training_data_quadratic)
    print()
    print("CUBIC:")
    print(training_data_cubic)
    print("TESTING:")
    print(test_data)

for i in range(0, 3):
    ensemble.append(generate_weight_set(2 + i))

''' y = "ax + b" '''
ensemble[0], total_error = train(training_data_linear, ensemble[0])
plot(training_data_linear, ensemble[0], "Training: y = ax + b          Training Error: " + str(float('%.3f'%(total_error))), "TrainingX")
error, predictions = test(test_data, ensemble[0])
plot(test_data, ensemble[0], "Testing: y = ax + b          Testing Error: " + str(float('%.3f'%(error))), "TestingX")

''' y = "ax^2 + bx + c" '''
ensemble[1], total_error = train(training_data_quadratic, ensemble[1])
plot(training_data_quadratic, ensemble[1], "Training: y = ax^2 + bx + c           Training Error: " + str(float('%.3f'%(total_error))), "TrainingX2")
error, predictions = test(test_data, ensemble[1])
plot(test_data, ensemble[1], "Testing: y = ax^2 + bx + c          Testing Error: " + str(float('%.3f'%(error))), "TestingX2")

''' y = "ax^3 + bx^2 + cx + d" '''
ensemble[2], total_error = train(training_data_cubic, ensemble[2])
plot(training_data_cubic, ensemble[2], "Training: y = ax^3 + bx^2 + cx + d          Training Error: " + str(float('%.3f'%(total_error))), "TrainingX3")
error, predictions = test(test_data, ensemble[2])
plot(test_data, ensemble[2], "Testing: y = ax^3 + bx^2 + cx + d          Testing Error: " + str(float('%.3f'%(error))), "TestingX3")


