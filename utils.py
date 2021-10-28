import pandas as pd
import numpy as np

categories = ["SCIENCE", "HISTORY", "LITERATURE", "MUSIC", "GEOGRAPHY"]


# TODO: Pensar em mais util functions, isto é super simples de implementar


def file_to_dataframe(filename, names=["Category", "Question", "Answer"]):
	assert filename is not None, TypeError("No filename given.")

	return pd.read_csv(filename, delimiter='\t', names=names)


"""
Get all lines from dataframe where column_name = column_value
"""
def get_fixed_column_lines(df, column_name, column_value):
	assert isinstance(df, pd.DataFrame), TypeError("Invalid 'df' parameter.")

	return df[df[column_name].str.contains(column_value)]


def get_row_by_number(df, row):
	assert isinstance(df, pd.DataFrame), TypeError("Invalid 'df' parameter.")
	assert row < len(df) and row >= 0, TypeError("Invalid row number.")

	return df.iloc[row]


def get_recall(guesses):
	assert isinstance(guesses, pd.DataFrame), TypeError("Invalid 'guesses' parameter.")
	assert np.array_equal(guesses.columns, ["Answer", "Correct Answer"]), TypeError("Invalid 'guesses' column names.")

	recall = {key: None for key in categories}

	for c in categories:
		# Get lines with Answer = c
		answers = get_fixed_column_lines(guesses, "Answer", c)

		# From the above lines, get the true positives (Correct Answer is also c)
		true_positives = sum(get_fixed_column_lines(answers, "Correct Answer", c))

		# From guesses get the total positives
		total_positives = sum(get_fixed_column_lines(guesses, "Correct Answer", c))

		recall[c] = true_positives/total_positives

	return recall




