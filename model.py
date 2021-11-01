from sklearn.metrics import jaccard_score


import metrics

def run_model_dev(path):
    real_categories = metrics.text_to_column_array(path, 0)
    questions = metrics.text_to_column_array(path, 1)
    #answers = metrics.text_to_column_array(path, 2)

    categories = naivebayes_model(questions,real_categories, questions)

    print(metrics.guess_accuracy(real_categories, categories))

def run_model_test(trainPath, testPath):
    trainCategories = metrics.text_to_column_array(trainPath, 0)
    trainQuestions = metrics.text_to_column_array(trainPath, 1)

    testCategories = metrics.text_to_column_array(testPath, 0)
    testQuestions = metrics.text_to_column_array(testPath, 1)

    categories = naivebayes_model(trainQuestions,trainCategories, testQuestions, preprocess=True)

    print(metrics.guess_accuracy(testCategories, categories))

    results_array = []
    for i in range(len(testCategories)):
        results_array.append([testCategories[i], categories[i]])

    print("Cats: GEO, HIST, LITERATURE, MUSIC, SCIENCE")
    print("Recall:", metrics.recall_from_array(results_array))
    print("Precision:", metrics.precision_from_array(results_array))
    print("FMeasure:", metrics.fmeasure_from_array(results_array))

    print("Jaccard: ", jaccard_score(testCategories, categories,average=None))

def naivebayes_model(questionsTrain, categoriesTrain, questionsTest, preprocess=True):
    q = metrics.sentences_to_array(questionsTrain)
    if(preprocess):
        qNotFiltered = metrics.remove_stopwords_from_array(questionsTrain)
        q = metrics.lemmatize_sentence_array(qNotFiltered)
        

    qc = []
    # Associate each array in list with a category
    for i in range(len(q)):
        qc.append([q[i],categoriesTrain[i]])

    # Calculate model for naive bayes

    # Start by calculating priors
    possible_cat = ["HISTORY","MUSIC","SCIENCE","GEOGRAPHY","LITERATURE"]
    priors = {}
    for cat in possible_cat:
        priors[cat] = 0

    for list in qc:
        for i in range(len(possible_cat)):
            if(list[1] == possible_cat[i]):
                priors[possible_cat[i]] += 1
                break
    
    for cat in possible_cat:
        priors[cat] = priors[cat] / len(qc)

    # Next calculate conditionals

    # Get all unique words in the train set
    words = []
    for list in qc:
        for w in list[0]:
            if w not in words:
                words.append(w)

    # calculates prior for every word for every cat
    cond = {}
    for word in words:
        for cat in possible_cat:

            appearences = 0
            totalWords = 0
            for l in qc:
                if(l[1] == cat):
                    totalWords += len(l[0])
                    for w in l[0]:
                        if(w == word):
                            appearences += 1
            val = (appearences+1) / (len(words) + totalWords)

            cond[word+cat] = val
            
    # test the model
    results = []


    # for every sentence in the test, we will turn it into an array, remove stop words, lemmatize
    # and check for the sum of the probabilities of the word

    t = metrics.sentences_to_array(questionsTest)
    if(preprocess):
        tNotFiltered = metrics.remove_stopwords_from_array(questionsTest)
        t = metrics.lemmatize_sentence_array(tNotFiltered)

    for s in t:
        chances = {}
        for c in possible_cat:
            chances[c] = priors[c]
            #chances[c] = 1
            for w in s:
                if(w+c in cond):
                    chances[c] *= cond[w+c]
        max = 0
        maxIndex = 0

        for i in range(len(possible_cat)):
            if(chances[possible_cat[i]] > max):
                max = chances[possible_cat[i]]
                maxIndex = i
        results.append(possible_cat[maxIndex])

    return results


run_model_test("./trainWithoutDev.txt","./dev_clean.txt")