import re
import Porter_Stemmer_Python as Porter_Stemmer
import math

CONST_SENTENCES_FILE = "Project4_sentences/sentences.txt"
CONST_STOP_WORDS_FILE = "Project4_sentences/stop_words.txt"
CONST_FREQUENCY_THRESHOLD = 3
CONST_LEARNING_CONTANT = 1
CONST_MAX_RADIUS = 3

def lines_into_list(file):
    # open file in read mode
    with open(file, 'r', encoding="utf-8") as file_handle:
        # convert file contents into a list remove new lines
        lines_list = file_handle.read().splitlines()
    return lines_list

def tokenize(list_of_sentences):
    tokenized_list = []
    for sentence in list_of_sentences:
        tokenized_sentence = sentence.split()
        tokenized_list.append(tokenized_sentence)
    return tokenized_list

def format_words(sentence_list):
    new_list = []
    pattern = r'[^A-Za-z ]'
    regex = re.compile(pattern)
    for sentence in sentence_list:
        if(len(sentence) != 0):
            new_sentence = []
            for word in sentence:
                 word = word.lower()
                 word = re.sub(regex, '', word)
                 new_sentence.append(word)
            new_list.append(new_sentence)
    return new_list

def remove_stop_and_empty(sentence_list, stop_words):
    new_list = []
    for sentence in sentence_list:
        if(len(sentence) != 0):
            new_sentence = []
            for word in sentence:
                if(len(word) != 0 and (word not in stop_words)):
                    new_sentence.append(word)
            new_list.append(new_sentence)
    return new_list

def stem_the_words(sentence_list):
    new_list = []
    stemmer = Porter_Stemmer.PorterStemmer()
    for sentence in sentence_list:
        if(len(sentence) != 0):
            new_sentence = []
            for word in sentence:
                if(len(word) != 0):
                    new_sentence.append(stemmer.stem(word, 0, (len(word)-1)))
            new_list.append(new_sentence)
    return new_list

def get_count_of_words(sentence_list):
    word_dict = {}
    #Gets the ccount of each word in all of the sentences
    for sentence in sentence_list:
        if(len(sentence) != 0):
            for word in sentence:
                if(word in word_dict):
                    word_dict[word] += 1
                else:
                    word_dict[word] = 1
    return word_dict

def extract_frequent_words(frequency_dict):
    most_frequent_words = []
    #Finds the words that occur more than frequent threshold
    for key in frequency_dict:
        if(frequency_dict[key] > CONST_FREQUENCY_THRESHOLD):
            most_frequent_words.append(key)
    for word in most_frequent_words:
        del frequency_dict[word]
    return frequency_dict

def extract_frequent_words_2(frequency_dict):
    least_frequent_words = []
    #Finds the words that occur more than frequent threshold
    for key in frequency_dict:
        if(frequency_dict[key] < CONST_FREQUENCY_THRESHOLD):
            least_frequent_words.append(key)
    for word in least_frequent_words:
        del frequency_dict[word]
    return frequency_dict

def initialize_sentences():
    all_sentences = lines_into_list(CONST_SENTENCES_FILE)
    stop_words = lines_into_list(CONST_STOP_WORDS_FILE)

    all_sentences = tokenize(all_sentences)
    all_sentences = format_words(all_sentences)
    all_sentences = remove_stop_and_empty(all_sentences, stop_words)

    return stem_the_words(all_sentences)

def create_tdm(sentences, word_dict_list):
    columns = len(word_dict_list)
    rows = len(sentences)
    tdm = [[0 for x in range(columns)] for y in range(rows)]
    #For each sentence
    for i in range(0, rows):
        if(len(sentences[i]) != 0):
            for j in range(0, len(sentences[i])): #For each word in the sentence
                if(len(sentences[i][j]) != 0 and (sentences[i][j] in word_dict_list)): #if sentence is in the word_dictionary add to  tdm
                    index_of_word = word_dict_list.index(sentences[i][j])
                    tdm[i][index_of_word] += 1
    return tdm

#Combines stemmed words by removing duplicates of stems within sentences
def combine_stemmed_words():
    sentences = initialize_sentences()
    temp = []
    for i in sentences:
       [temp.append(x) for x in i if x not in temp]
       sentences[sentences.index(i)] = temp
       temp = []

    return sentences

def calculate_distance(list1, list2):
    distance = 0
    for i in range(0, len(list1)):
        difference = (float(list1[i]) - float(list2[i]))
        distance += math.pow(difference, 2)
    return math.sqrt(distance)

def update_COG(current_cog, newest_points):
    num_of_clusters = current_cog[1]
    for i in range(0, len(current_cog[0])):
        current_cog[0][i] = ((current_cog[0][i] * num_of_clusters) + (CONST_LEARNING_CONTANT * newest_points[i])) / (num_of_clusters + 1)
    return current_cog

def form_clusters(tdm):
    num_of_clusters = 1
    clusters = [[tdm[0], 1]]    #Initializes the first cluster
    for i in range(1, len(tdm)):
        distances = []
        for j in range(0, len(clusters)):
            distance = calculate_distance(clusters[j][0], tdm[i])
            distances.append(distance)
        minimum = min(distances)
        if(minimum < CONST_MAX_RADIUS):
            index = distances.index(minimum)
            clusters[index] = update_COG(clusters[index], tdm[i])
            clusters[index][1] += 1
        else:
            clusters.append([tdm[i], 1])
    return clusters

all_sentences = initialize_sentences()

num_words_dict = get_count_of_words(all_sentences)
num_words_dict = extract_frequent_words_2(num_words_dict)


keys_list = [key for key in num_words_dict] #Generates an ordered list of the keys.
''' TDM is in format: Rows = sentences .. Columns = words
    TDM[sentence][word] value is the frequency of a word in a given sentence '''

print("FEATURE VECTOR")
print(keys_list)
tdm = create_tdm(all_sentences, keys_list)

print("TDM")
print(tdm)

f = open("tdm.csv", "x")
for item in tdm:
    for entry in item:
        f.write("{},".format(entry))
    f.write("\n")

all_clusters = form_clusters(tdm)


print("NUMBER OF CLUSTERS: " + str(len(all_clusters)))
for cluster in all_clusters:
    print(cluster)
    print()

'''
test_list = [[5.9630, 0.7258], [4.1168, 2.9694], [1.8184, 6.0148], [6.2139, 2.4288],
[6.1290, 1.3876], [1.0562, 5.8288], [4.3185, 2.3792], [2.6108, 5.4870], [1.5999, 4.1317],
[1.1046, 4.1969]]

values = form_clusters(test_list)
print("NUMBER OF CLUSTERS: " + str(len(values)))
print(values)
'''
