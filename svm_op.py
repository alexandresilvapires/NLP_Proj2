from sys import argv

import nltk
from nltk.corpus import stopwords

from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import jaccard_score
from sklearn.feature_extraction.text import TfidfVectorizer

import utils



"""
Convert arguments to panda dataframes,
and create new column 'Content'
"""
def convert_to_pandas(train_set_filename, test_set_filename):
	train_df = utils.file_to_dataframe(train_set_filename)
	test_df = utils.file_to_dataframe(test_set_filename)

	# 'Content' = 'Question' + 'Answer'
	train_df['Content'] = train_df['Question'] + ' ' + train_df['Answer'] + ' ' + train_df['Answer']
	train_df['Content'] = train_df['Content'].fillna('') # replace NaN with ''
	
	test_df['Content'] = test_df['Question'] + ' ' + test_df['Answer']
	test_df['Content'] = test_df['Content'].fillna('')

	return train_df, test_df


"""
Split datasets into t ('Content' list) and c ('Category' list)
"""
def split_data(train_df, test_df):
	train_t = train_df['Content'].tolist()
	train_c = train_df['Category'].tolist()

	test_t = test_df['Content'].tolist()
	test_c = test_df['Category'].tolist()

	return train_t, train_c, test_t, test_c


"""
Preprocess and vectorize arguments
"""
def vectorize(train_t, test_t):
	stop_words = set(stopwords.words("english"))
	
	
	vectorizer = TfidfVectorizer(analyzer='word',smooth_idf=False, lowercase=True, stop_words=stop_words)

	train_t = vectorizer.fit_transform(train_t)
	test_t = vectorizer.transform(test_t)


	return train_t, test_t

"""
Vectorize arguments
"""
def vectorize_no_preprocessing(train_t, test_t):

	vectorizer = TfidfVectorizer(analyzer='word')

	train_t = vectorizer.fit_transform(train_t)
	test_t = vectorizer.transform(test_t)


	return train_t, test_t

def main():
	# check arguments
	if len(argv) != 3:
		raise ValueError("Correct argument format: program train_set_filename test_set_filename")

	# download stopwords set
	nltk.download('stopwords', quiet=True)

	# text preprocessing
	train_df, test_df = convert_to_pandas(argv[1], argv[2])
	train_t, train_c, test_t, test_c = split_data(train_df, test_df)
	train_t, test_t = vectorize(train_t, test_t)

	# build and train NAIVE BAYES
	print("Building and training NB...")
	nb = MultinomialNB().fit(train_t, train_c)
	print("Done!")

	# test naive bayes
	prediction = nb.predict(test_t)
	accuracy = accuracy_score(test_c, prediction)
	print(classification_report(test_c, prediction))
	print("Jaccard score:" , jaccard_score(test_c, prediction,average=None))
	print("Naive Bayes accuracy: {}%".format(accuracy*100))


	# build and train SUPPORT VECTOR MACHINE
	print("\nBuilding and training SVM...")
	#svm = SGDClassifier(max_iter=20000, tol=1e-5,alpha=0.00006).fit(train_t, train_c)

	svm = LinearSVC().fit(train_t, train_c)

	print("Done!\n")

	# test svm
	prediction = svm.predict(test_t)
	accuracy = accuracy_score(test_c, prediction)
	print(classification_report(test_c, prediction))
	print("Jaccard score:" , jaccard_score(test_c, prediction,average=None))
	print("SVM accuracy: {}%".format(accuracy*100))

if __name__ == '__main__':
	main()