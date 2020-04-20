import pandas as pd
import random
import math
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

CONST_NUMBER_OF_NEURONS = 3
CONST_LEARNING_CONSTANT = .5

#Graphs wta for a given set of patterns and weights
def graph_wta(patterns, weights, graph_title):
    fig, ax = plt.subplots(figsize=(8,7), dpi=100)
    for points in patterns:
        m1 = plt.plot(points[0], points[1], color="red", marker='o', markersize=5, alpha=.5)
    for weight in weights:
        m2 = plt.plot(weight[0], weight[1], color="blue", marker='>', markersize=6)
    plt.title(graph_title)
    if(len(weights) > 0):
        legend_elements = [Line2D([0], [0], marker='o', color='red', label='Data', markersize=6),
        Line2D([0], [0], marker='>', color='blue', label='Neuron', markersize=7),]
    else:
        legend_elements = [Line2D([0], [0], marker='o', color='red', label='Data', markersize=6)]
    plt.legend(handles=legend_elements, loc='upper right')
    file_name = "Results/" + str(CONST_NUMBER_OF_NEURONS) + "Neurons/" +graph_title + "2.png"
    plt.savefig(file_name)
    plt.show()

# Number of neurons will be number of weight vectors
# number_of_weights will be the number of weights in each weight vector
def assign_random_weights(number_of_weights, val=0, ranges=[]):
    weights = []
    for i in range(0, CONST_NUMBER_OF_NEURONS):
        random_index = random.randrange(0, len(patterns))
        if(val == 0):
            new_list = []
            for j in range(0, number_of_weights):
                new_list.append(random.random())
            weights.append(new_list)
        else:
            new_list = []
            new_list.append(random.randrange(round(ranges[0][0]), -1))
            new_list.append(random.randrange(round(ranges[1][0]), round(ranges[1][1])))
            weights.append(new_list)
    return weights

# Normalizes to "unity"
def normalize(list_to_normalize):
    new_list = []
    total_sum_square = 0
    for element in list_to_normalize:
        total_sum_square += math.pow(element , 2)
    for element in list_to_normalize:
        if(total_sum_square == 0):
            total_sum_square = 1
        element = round((element / (math.sqrt(total_sum_square))), 4)
        new_list.append(element)
    return new_list

#Calculates the net for a given pattern and neuron
def calculate_net(pattern, neuron):
    net_sum = 0
    for i in range(0, len(pattern)):
        net_sum += round((pattern[i] * neuron[i]), 4)
    return round(net_sum,4)

#Updates the winning neuron
def update_neuron(pattern, neuron):
    for n in range(0, len(neuron)):
        neuron[n] = round((neuron[n] + (CONST_LEARNING_CONSTANT * pattern[n])), 4)
    return neuron

#Calculates nets for each neuron on each pattern. Winning neuron gets weights updated
def winner_take_all(patterns, neurons, iteration=1):
    count = 0
    for pattern in patterns:
        all_nets = []
        for neuron in neurons:
            net = calculate_net(pattern, neuron)
            all_nets.append(net)
        max_value = max(all_nets)
        max_index = all_nets.index(max_value) #index of the winning neuron
        neurons[max_index] = update_neuron(pattern, neurons[max_index])
        neurons[max_index] = normalize(neurons[max_index])
        count+=1
        if(iteration == 1 and count<5):
            graph_wta(patterns, neurons, "Iteration " + str(iteration) + " Pattern " + str(count))
        #print("UPDATED NEURON" + str(max_index))
        #print(neurons[max_index])
    return neurons

#Reads data
data = pd.read_csv("Data/Ex1_data.csv")

#Manually set min and max based on data
min_max_x_y = [[-11.7, .91], [-10.38, 8.1]]

patterns = data.values.tolist()
#Assigns random weights within the range of the data.
weights = assign_random_weights(2, 1, min_max_x_y)

for w in range(0, len(weights)):
    weights[w] = normalize(weights[w])

num = 0
graph_wta(patterns, [], "Initial Data")
for p in range(0, len(patterns)):
    patterns[p]  = normalize(patterns[p])

graph_wta(patterns, weights, "Normalized Patterns and Neurons")

for i in range(0,1):
    num += 1
    weights = winner_take_all(patterns, weights)
    print(weights)

graph_wta(patterns, weights, "After 1 Iteration")

for i in range(0,4):
    num += 1
    weights = winner_take_all(patterns, weights, 2)
    print(weights)

graph_wta(patterns, weights, "After 5 Iterations")

for i in range(0,5):
    num += 1
    weights = winner_take_all(patterns, weights, 2)
    print(weights)

graph_wta(patterns, weights,"After 10 Iterations")


# Test weights and patterns already normalized to unity
test_weights = [[.9459, .3243], [.669, .7433], [.3714, .9285]]
test_patterns = [[0.9927, 0.1208], [0.8110, 0.5850], [0.2894, 0.9572], [0.9314, 0.3640], [0.9753, 0.2208],
[0.1783, 0.9840], [0.8759, 0.4825], [0.4296, 0.9030], [0.3611, 0.9325], [0.2545, 0.9671]]

''' Test produces expected numbers as per session 17
for i in range(0, 30):
    test_weights = winner_take_all(test_patterns, test_weights)
    print("ITERATION " +  str(i))
    print(test_weights)
'''

'''For a 2d list like above send 1 pattern at a time to normalize.
for l in range(0, len(test)):
    test[l] = normalize(test[l])
'''
