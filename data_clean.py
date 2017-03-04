import numpy as np
import pandas
import os

def data_path():
	parent_path = os.path.pardir
	data_path = parent_path + "/TripAdvisorChallenge/datathon_tadata.csv"
	return data_path
