import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer

"""
Downloads every necessary lib from nltk
"""
def download_resources():
    nltk.download('stopwords')
    nltk.download('punkt')
    nltk.download('wordnet')

"""
Given a text file, extracts a column to an array
"""
def text_to_column_array(path, colNumber):
    file = open(path,"rt")

    lines = file.readlines()
    col = []

    for line in lines:
        columns = line.split("\t")
        col.append(columns[colNumber])

    return col
        

"""
Given an array of lists with the format (guess, correct cat.), creates an agreement matrix 
with entries GEOGRAPHY, MUSIC, LITERATURE, HISTORY, SCIENCE, where columns are the correct answers
"""
def agreement_matrix(guesses):

    def string_to_index(cat):
        if(cat == "GEOGRAPHY"):
            return 0
        elif(cat == "MUSIC"):
            return 1
        elif(cat == "LITERATURE"):
            return 2
        elif(cat == "HISTORY"):
            return 3
        else:
            return 4

    matrix =   [[0,0,0,0,0],
                [0,0,0,0,0],
                [0,0,0,0,0],
                [0,0,0,0,0],
                [0,0,0,0,0]]

    for guess in guesses:
        i = string_to_index(guess[0])
        j = string_to_index(guess[1])
        matrix[i][j] += 1

    return matrix


"""
Given an array of lists with the format (guess, correct cat.), returns the precision of each category
That is, an array with the precision for GEOGRAPHY, MUSIC, LITERATURE, HISTORY, SCIENCE
"""
def precision_from_array(guesses):

    # Get the agreement matrix
    matrix = agreement_matrix(guesses)

    # Calculate precision for each category
    catPrecision = [1.0,1.0,1.0,1.0,1.0]
    for i in range(0,5):
        truePos = matrix[i][i]
        falsePos = 0
        
        for j in range(0,5):
            if(i != j):
                # The false positives are given by the sum of the line of the guess category except the right one
                falsePos += matrix[i][j]
        
        if(truePos + falsePos != 0):
            catPrecision[i] = truePos / (truePos + falsePos)

    return catPrecision

"""
Given an array of lists with the format (guess, correct cat.), returns the recall of each category
That is, an array with the precision for GEOGRAPHY, MUSIC, LITERATURE, HISTORY, SCIENCE
"""
def recall_from_array(guesses):

    # Get the agreement matrix
    matrix = agreement_matrix(guesses)

    # Calculate precision for each category
    catPrecision = [1.0,1.0,1.0,1.0,1.0]
    for i in range(0,5):
        truePos = matrix[i][i]
        falseNeg = 0
        
        for j in range(0,5):
            if(i != j):
                # The false negatives are given by the sum of the line of the guess category except the right one
                falseNeg += matrix[j][i]
        
        if(truePos + falseNeg != 0):
            catPrecision[i] = truePos / (truePos + falseNeg)

    return catPrecision

"""
Given an array of lists with the format (guess, correct cat.), returns the F measure of each category
That is, an array with the F measure for GEOGRAPHY, MUSIC, LITERATURE, HISTORY, SCIENCE
"""
def fmeasure_from_array(guesses, b = 1):

    # Gets the precision and the recall
    precision = precision_from_array(guesses)
    #recall = recall_from_array(guesses)
    recall = [1,1,1,1,1]

    # Calculate F for each category
    fmeasure = [1,1,1,1,1]
    for i in range(0,5):
        fmeasure[i] = ((b * b + 1) * precision[i] * recall[i] ) / (b * b * precision[i] + recall[i])

    return fmeasure


def average_from_array(array):
    av = 0
    for val in array:
        av += val
    return av / len(array)

"""
Given an array of sentences, returns an array of arrays of words after filtering stop words
"""
def sentences_to_array(sentences):
    newSentences = []
    stopw = set(stopwords.words("english"))

    for s in sentences:
        tokenizer = RegexpTokenizer(r'\w+')
        word_tokens = tokenizer.tokenize(s)
        newSentences.append(word_tokens)

    return newSentences


"""
Given an array of sentences, returns an array of arrays of words after filtering stop words
"""
def remove_stopwords_from_array(sentences):
    newSentences = []
    stopw = set(stopwords.words("english"))

    for s in sentences:
        tokenizer = RegexpTokenizer(r'\w+')
        word_tokens = tokenizer.tokenize(s)

        filtered_sentence = [w for w in word_tokens if not w.lower() in stopw]
        newSentences.append(filtered_sentence)

    return newSentences

"""
Given an array of arrays (each word of each sentence), returns the array with each word lemmatized
"""
def lemmatize_sentence_array(sentences):
    newSentences = []
    lemmatizer = WordNetLemmatizer()

    for s in sentences:
        newS = []
        
        for w in s:
            newS.append(lemmatizer.lemmatize(w))
        newSentences.append(newS)
    return newSentences

"""
Given two arrays, one with the correct categories, and one with the guesses, returns an accuracy value between 0 and 1
"""
def guess_accuracy(correct, guess):
    rights = 0
    for i in range(len(correct)):
        if(correct[i] == guess[i]):
            rights += 1

    return rights/len(correct)
