import ipdb
import numpy as np
import pandas as pd
import os
from collections import Counter
from sklearn.preprocessing import MinMaxScaler

def get_data_path():
	'''
	return the path of the data to load into a numpy array.
	'''
	parent_path = os.path.pardir
	data_path = parent_path + "/TripAdvisorChallenge/datathon_tadata.csv"
	return data_path

def date_counter(dates):
	'''
	determine what dates we're dealing with; dates is a vector
	'''
	c = Counter(dates)
	lst = list(c.most_common())
	print(lst)

def clean_data():
	'''
	clean the data to do machine learning.
	'''
	data = pd.read_csv(get_data_path()).as_matrix()
	date_counter(data[:,1])
	num_examples = data.shape[0]
	# ipdb.set_trace()  ######### Break Point ###########

	# save the columns to be removed. p_traffic_channel_column and operating_system_column have non-numerical data so we will rearrange these things.
	labels_column = data[:, 16]
	p_traffic_channel_column = data[:, 5]
	operating_system_column = data[:, 10]

	# sparse matrices
	p_traffic_channel_matrix = map_to_matrix(p_traffic_channel_column)
	operating_system_matrix = map_to_matrix(operating_system_column)

	# remove non-numerical data
	data = np.delete(data, [5, 10, 16], 1)
	data = np.concatenate((data, p_traffic_channel_matrix, operating_system_matrix), 1)

	# set leftmost column to the labels.
	data = np.concatenate((np.reshape(labels_column, (num_examples, 1)), data), 1)

	# remove user id and the date.
	# TODO: try to incorporate the date somehow as a feature.
	data = np.concatenate((np.reshape(data[:, 0], (num_examples, 1)), data[:, 3:]), 1)

	# interpolate
	dff = pd.DataFrame(data)
	dff = dff.fillna(dff.mean())

	# normalize data set_trace
	scaler = MinMaxScaler(feature_range=(0, 1))
	dff = scaler.fit_transform(dff)
	return dff

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
	np.save('cleaned_data', data)
	# print(np.load('cleaned_data.npy'))
	# print(data.as_matrix()[0, :])
