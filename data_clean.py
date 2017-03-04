import numpy as np
import pandas
import os
from collections import Counter

def get_data_path():
	'''
	return the path of the data to load into a numpy array.
	'''
	parent_path = os.path.pardir
	data_path = parent_path + "/TripAdvisorChallenge/datathon_tadata.csv"
	return data_path

def clean_data():
	'''
	clean the data to do machine learning.
	'''
	# print(get_data_path())
	data = pandas.read_csv(get_data_path()).as_matrix()
	# operating_system_column = data[:, 10]
	# c = Counter(operating_system_column)
	# print(list(c.most_common()))
	return data

def map_to_matrix(iterable):

	'''
	consumes an iterable and use collections.Counter to return a matrix of numbers according to the class.

	Example
	input:
	['windows',
	'mac',
	'linux',
	'mac',
	'linux',
	'linux']

	return: an n x m matrix, n being the length of the original iterable and m
	being the number of unique classes--in this example, n = 6 and m = 3.

	Let the ith row, jth column element be 1 if the ith element in the input
	belongs to class j. Otherwise make everything 0. Let the ordering of the classes be such that the keys are in order of greatest to least frequency.

	Thus in this case, class 0 = linux, class 1 = mac, class 2 = windows.

	Then the return value should be a numpy matrix as follows:

	0	0	1
	0	1	0
	1	0	0
	0	1	0
	1	0	0
	1	0	0
	'''
	c = Counter(iterable)
	n = len(iterable)
	m = len(c)
	k_list = list(c.most_common()) # key-frequency list
	k = [x for (x, y) in k_list] # list of keys
	matrix = np.zeros([n,m])

	for index, example in enumerate(iterable):
		# TODO: try to optimize this lookup problem, probably with hashtables or something.
		class_number = k.index(example)
		matrix[index, class_number] = 1
	return matrix

if __name__ == '__main__':
	data = clean_data()
	operating_system_column = data[:, 10]
	x = map_to_matrix(operating_system_column)
	print(sum(x)[0])
	# print(sum(sum(map_to_matrix(operating_system_column))))


	# print(data[0, 1]) # should be 2017-01-10
