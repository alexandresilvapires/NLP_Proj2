import metrics

def run_model_dev(path):

    real_categories = metrics.text_to_column_array(path, 0)
    questions = metrics.text_to_column_array(path, 1)
    answers = metrics.text_to_column_array(path, 2)

    #categories = run_model = run_model(questions, answers)

    #print(metrics.guess_accuracy(real_categories, categories))
