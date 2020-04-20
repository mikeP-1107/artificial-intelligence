import pandas as pd
import math
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D
CONST_CLASS_INDEX = 1

#load the file into dataframe and return it a a list
def load_file(filename): #take filename as param
    data = pd.read_csv(filename)
    for i in range(len(data)):
        data.iloc[i] = [float(x) for x in data.iloc[i]] #turn all into floats for calculation prep
    return data.values.tolist()

# Take in the dictionary of split classes return total number of objects across all classes.
def get_total(data):
    total = 0
    for key in data:
        total += data[key]
    return total

# Take in dictionary of split classes. Return dictionary of number of objects in each class
def get_class_counts(data):
    class_lists = {}
    for val in data:
        if val[CONST_CLASS_INDEX] in class_lists:
            class_lists[val[CONST_CLASS_INDEX]] += 1
        else:
            class_lists[val[CONST_CLASS_INDEX]] = 1
    return class_lists

# Split the dataset by class values, returns a dictionary
def divide_classes(data):
	class_lists = dict()
	for i in range(len(data)):
		current_val = data[i]
		class_value = current_val[-1]
		if (class_value not in class_lists):
			class_lists[class_value] = list()
		class_lists[class_value].append(current_val[0])
	return class_lists

# Take in dictionary of split classes. Return dictionary of mean of each class
def calculate_mean(data):
    means = {}
    for key in data:
        means[key] = 0
        for val in data[key]:
            means[key] += val
        means[key] /= (len(data[key]))
    return means

# Take in dictionary of split classes. Return dictionary of standard deviation of each class
def calculate_deviations(data, means):
    standard_deviations = {}
    for key in data:
        standard_deviations[key] = 0
        for val in data[key]:
            standard_deviations[key] += math.pow((val - means[key]) , 2)
        standard_deviations[key] /= (len(data[key]) - 1)
        standard_deviations[key] = math.sqrt(standard_deviations[key])
    return standard_deviations

#Take in testdata, standard_deviations, and means. Returns calculated  P(x|ci)
def bayes_calculate_probs_given_x(testing_data, std_dv, means):
    all_probabilities = []
    for sample in testing_data:
        probabilities = {}
        current_prob = 0
        for key in std_dv:
            current_prob = math.exp(-(math.pow((sample - means[key]) , 2)) / (2 * math.pow(std_dv[key] , 2)))
            probabilities[key] = (current_prob * ((1)/((std_dv[key]) * math.sqrt(2 * math.pi))))  #P(sample|C)
        all_probabilities.append(probabilities)
    return all_probabilities

#Take in data. Return calculated Probabilities P(ci)
def bayes_p_of_x(data):
    total_counts = get_total(data)
    probs = {}
    for key in data:
        probs[key] = ((data[key])/total_counts)
    return probs
#Takes in p(ci) and p(x|ci) and multiplies them together. Returns result
def bayes_p_times_p_given_x(p_of_x, p_given_x):
    bayes_p_times_p_given_x = []
    for probability_dict in p_given_x:
        all_vals = {}
        for key in probability_dict:
            all_vals[key] = (probability_dict[key] * p_of_x[key])
        bayes_p_times_p_given_x.append(all_vals)
    return bayes_p_times_p_given_x

#Takes in p(ci)*p(x|ci) results and returns the predictions based on max value
def bayes_prediction(all_probabilities):
    final_predictions = []
    for probability_dict in all_probabilities:
        max = float('-inf')
        max_key = "placeholder"
        for key in probability_dict:
            if (probability_dict[key] > max):
                max_key = key
                max = probability_dict[key]
        final_predictions.append({max_key: max})
    return final_predictions

#Takes in p(ci), p(x|ci) and p(ci)*p(x|ci) puts all results out to csvs
def probabilities_to_text(p_of_x, p_given_x, p_times_p_given_x):
    keys = []
    for key in p_of_x:
        keys.append("Class " + str(key))
    p_of_x_df = pd.DataFrame(p_of_x.values(), index = keys)
    p_of_x_df.to_csv("Results/p(ci).csv")
    p_given_x_df = pd.DataFrame(p_given_x)
    p_given_x_df.columns = keys
    p_given_x_df.to_csv("Results/p(x|ci).csv")
    p_times_p_given_x_df = pd.DataFrame(p_times_p_given_x)
    p_times_p_given_x_df.columns = keys
    p_times_p_given_x_df.to_csv("Results/p(x|ci)_x_p(ci).csv")

#Takes in testing data and predictions, Calculates the accuracy of predictions and puts them to csv
def test_accuracy(testing_data, predictions):
    total = len(testing_data)
    correct = 0
    zeros = [0, 0, 0]
    ones = [0, 0, 0]
    twos = [0, 0, 0]

    for i in range(len(predictions)):
        for key in predictions[i]:
            if key == testing_data[i][1]:
                correct+=1
            if key == 0:
                zeros[int(testing_data[i][1])] += 1
            elif key == 1:
                ones[int(testing_data[i][1])] += 1
            else:
                twos[int(testing_data[i][1])] += 1
    confusion_matrix = [zeros, ones, twos]
    confusion_matrix = pd.DataFrame(confusion_matrix)
    confusion_matrix.to_csv("Results/ConfusionMatrix.csv")

#Load Training and testing data
training_data = load_file("Data/Ex2_train.csv")
testing_data = load_file("Data/Ex2_test.csv")

#Removes class definition for making predictions
testing_data_no_class = [row[0] for row in testing_data]

#Split training_data and testing data by class
class_instances  = divide_classes(training_data)
testing_class_instances = divide_classes(testing_data)

#Calculate means, standard_deviations and counts of training_data
means = calculate_mean(class_instances)
counts = get_class_counts(training_data)
standard_deviations = calculate_deviations(class_instances, means)

#Calculate p(x|ci), p(ci), and p(i) * p(x|ci)
p_given_x = bayes_calculate_probs_given_x(testing_data_no_class, standard_deviations, means)
p_given_xdf = pd.DataFrame(p_given_x)
p_of_x = bayes_p_of_x(counts)
p_times_p_given_x = bayes_p_times_p_given_x(p_of_x, p_given_x)

#Get predictions
predictions = bayes_prediction(p_times_p_given_x)
x = [standard_deviations[key] for key in standard_deviations]

#Test the accuray of the predications made
test_accuracy(testing_data, predictions)

#Output all necessary info to csvs
probabilities_to_text(p_of_x, p_given_x, p_times_p_given_x)

#Plot data and distribution of data
fig, ax = plt.subplots(figsize=(7,6), dpi=100)
sns.distplot(class_instances[0], hist=False, color="blue", label=0)
sns.distplot(class_instances[1], hist=False, color="red", label=1)
sns.distplot(class_instances[2], hist=False, color="purple", label=2)
for key in testing_class_instances:
    if(key == 0):
        color = "blue"
    elif(key == 1):
        color = "red"
    else:
        color = "purple"
    for point in testing_class_instances[key]:
        plt.plot(point, 0, color=color, marker='o', markersize=6, alpha=.7)
plt.title('Data Distributions')

legend_elements = [Line2D([0], [0], marker='o', color='blue', label='Class 0 Test Data Point', markersize=7),
Line2D([0], [0], color='blue', lw=4, label='Class 0 Distribution'),
Line2D([0], [0], marker='o', color='red', label='Class 1 Test Data Point', markersize=6),
Line2D([0], [0], color='red', lw=4, label='Class 1 Distribution'),
Line2D([0], [0], marker='o', color='purple', label='Class 2 Test Data Point', markersize=6),
Line2D([0], [0], color='purple', lw=4, label='Class 2 Distribution'),]

ax.legend(handles=legend_elements)
file_name = "Results/" + "Data_Distribution" + ".png"
plt.savefig(file_name)
plt.show()


'''
test_instances = { 0: [56.0307, 64.2989, 60.3343, 51.8031, 47.8763, 58.5809, 65.7290, 60.9058, 60.2713, 63.4388]
, 1: [73.0330, 87.1268, 75.5307, 80.1888, 78.1822, 80.7478, 70.2774, 87.6195, 82.7291, 90.0496, 87.0834, 80.0574,
75.3048, 71.3055, 80.0849]}d
test_means = calculate_mean(test_instances)
print("TEST MEANS")
print(test_means)
test_deviations = calculate_deviations(test_instances, test_means)
print("TEST STANDARD DEVIATIONS")
print(test_deviations)
test_counts = { 0: 10, 1: 15}
test_samples_given_x = bayes_calculate_probs_given_x([75], test_deviations, test_means)
print("TEST P GIVEN X")
print(test_samples_given_x)
test_std_probability = bayes_p_of_x(test_counts)
print("TEST STANDARD PROBABILITY")
print(test_std_probability)
test_p_times_p_given_x = bayes_p_times_p_given_x(test_std_probability, test_samples_given_x)
print("TEST P TIMES P GIVEN X")
print(test_p_times_p_given_x)
test_predictions = bayes_prediction(test_p_times_p_given_x)
print("TEST PREDICTIONS")
print(test_predictions)
'''
