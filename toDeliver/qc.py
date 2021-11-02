import argparse

import nltk
from nltk.corpus import stopwords

from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer

import numpy as np
import pandas as pd


def file_to_dataframe(filename, names=["Category", "Question", "Answer", "Noise"]):
	assert filename is not None, TypeError("No filename given.")

	return pd.read_csv(filename, delimiter='\t', names=names)


"""
Convert arguments to panda dataframes,
and create new column 'Content'
"""
def convert_to_pandas(train_set_filename, test_set_filename):
	train_df = file_to_dataframe(train_set_filename)
	test_df = file_to_dataframe(test_set_filename, names =["Question","Answer","Noise"])

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

	return train_t, train_c, test_t


"""
Preprocess and vectorize arguments
"""
def vectorize(train_t, test_t):
	stop_words = set(stopwords.words("english"))
	
	vectorizer = TfidfVectorizer(analyzer='word',smooth_idf=False, lowercase=True, stop_words=stop_words)

	train_t = vectorizer.fit_transform(train_t)
	test_t = vectorizer.transform(test_t)

	return train_t, test_t


def main():
	# get arguments
	parser = argparse.ArgumentParser()
	parser.add_argument("-test", "--test", help="Test Filename", type=str)
	parser.add_argument("-train", "--train", help="Train Filename", type=str)
	args = parser.parse_args()

	# download stopwords set
	nltk.download('stopwords', quiet=True)

	# text preprocessing
	train_df, test_df = convert_to_pandas(args.train, args.test)
	train_t, train_c, test_t = split_data(train_df, test_df)
	train_t, test_t = vectorize(train_t, test_t)

	# build and train SUPPORT VECTOR MACHINE
	svm = LinearSVC().fit(train_t, train_c)

	# test svm
	prediction = svm.predict(test_t)
	for p in prediction:
		print(p)


if __name__ == '__main__':
	main()