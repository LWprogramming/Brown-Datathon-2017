import numpy as np
import pandas
import os

def get_data_path():
	parent_path = os.path.pardir
	data_path = parent_path + "/TripAdvisorChallenge/datathon_tadata.csv"
	return data_path

def clean_data():
	# print(get_data_path())
	data = pandas.read_csv(get_data_path()).as_matrix()
	return data

if __name__ == '__main__':
	data = clean_data()
	print(data[0, 1]) # should be 2017-01-10
