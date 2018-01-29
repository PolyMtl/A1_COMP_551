 # test
import csv
import pandas as pd
import numpy as np

def read_csv(file_name):
	df = pd.read_csv(file_name, encoding='utf-8', header = None, comment='#', sep=',', na_values='?')
	return df

def convert_to_csv(file_name, df):
	df.to_csv(file_name, encoding='utf-8', header = None, sep=',')

def fill_missing_values(df):
	# predictive columns start from the 5th column onwards
	null_count = 0
	for column in df.columns[5:]:
		mean = df[column].mean()
		std_dev = df[column].std()
		null_count = df[column].isnull().sum()

		if(mean - std_dev) < 0:
			null_list = np.random.uniform(0,2*mean, size = null_count)
		else:
			null_list = np.random.uniform(mean - std_dev,mean + std_dev, size = null_count)

		# this line gives warning (SettingWithCopyWarning) but fills missing values as wanted
		df[column][np.isnan(df[column])] = null_list

	return df

def create_input_matrix(dataset):
	# adding a column of ones
	dataset.insert(0,'',1)
	dataset.column = [np.arange(0, dataset.shape[1])]
	dataset = dataset.drop(dataset.index[123])
	# convert to array for ease of use
	dataset =  np.array(dataset)
	return dataset

def create_output_vector(dataset):
	dataset = dataset[126]
	return np.array(dataset)


if __name__ == "__main__":
	# 3.1
	# get dataset_3
	df = read_csv(r'/Users/vivek/git/COMP_551_A1/Datasets/Dataset_3.csv')
	df
	# fill in the missing values 
	df_modified = fill_missing_values(df)
	# convert the new dataset into csv
	convert_to_csv(r'/Users/vivek/git/COMP_551_A1/Datasets/Dataset_3_modified.csv', df_modified)

	# 3.2 (80-20 datasplits)
	# create a training dataset (80% of whole dataset)
	training_set_1 = df_modified.sample(frac = 0.8, random_state = 100)
	# remaining is test set
	test_set_1 = df_modified.drop(training_set_1.index)

	# drop the first 5 non predictive columns
	training_set_1 = training_set_1.drop(range(5), axis=1)
	test_set_1 = test_set_1.drop(range(5), axis=1)

	# convert the new dataset into csv
	convert_to_csv(r'/Users/vivek/git/COMP_551_A1/Datasets/CandC_train_1.csv', training_set_1)
	convert_to_csv(r'/Users/vivek/git/COMP_551_A1/Datasets/CandC_test_1.csv', test_set_1)

	# repeat this 4 more times --> total of 5 train/test datasets of 80-20 split
	# create a training dataset (80% of whole dataset)
	training_set_2 = df_modified.sample(frac = 0.8, random_state = 100)
	# remaining is test set
	test_set_2 = df_modified.drop(training_set_2.index)

	# drop the first 5 non predictive columns
	training_set_2 = training_set_2.drop(range(5), axis=1)
	test_set_2 = test_set_2.drop(range(5), axis=1)

	# convert the new dataset into csv
	convert_to_csv(r'/Users/vivek/git/COMP_551_A1/Datasets/CandC_train_2.csv', training_set_2)
	convert_to_csv(r'/Users/vivek/git/COMP_551_A1/Datasets/CandC_test_2.csv', test_set_2)
	
	# create a training dataset (80% of whole dataset)
	training_set_3 = df_modified.sample(frac = 0.8, random_state = 100)
	# remaining is test set
	test_set_3 = df_modified.drop(training_set_3.index)

	# drop the first 5 non predictive columns
	training_set_3 = training_set_3.drop(range(5), axis=1)
	test_set_3 = test_set_3.drop(range(5), axis=1)

	# convert the new dataset into csv
	convert_to_csv(r'/Users/vivek/git/COMP_551_A1/Datasets/CandC_train_3.csv', training_set_3)
	convert_to_csv(r'/Users/vivek/git/COMP_551_A1/Datasets/CandC_test_3.csv', test_set_3)

	# create a training dataset (80% of whole dataset)
	training_set_4 = df_modified.sample(frac = 0.8, random_state = 100)
	# remaining is test set
	test_set_4 = df_modified.drop(training_set_4.index)

	# drop the first 5 non predictive columns
	training_set_4 = training_set_4.drop(range(5), axis=1)
	test_set_4 = test_set_4.drop(range(5), axis=1)

	# convert the new dataset into csv
	convert_to_csv(r'/Users/vivek/git/COMP_551_A1/Datasets/CandC_train_4.csv', training_set_4)
	convert_to_csv(r'/Users/vivek/git/COMP_551_A1/Datasets/CandC_test_4.csv', test_set_4)

	# create a training dataset (80% of whole dataset)
	training_set_5 = df_modified.sample(frac = 0.8, random_state = 100)
	# remaining is test set
	test_set_5 = df_modified.drop(training_set_5.index)

	# drop the first 5 non predictive columns
	training_set_5 = training_set_5.drop(range(5), axis=1)
	test_set_5 = test_set_5.drop(range(5), axis=1)

	# convert the new dataset into csv
	convert_to_csv(r'/Users/vivek/git/COMP_551_A1/Datasets/CandC_train_5.csv', training_set_5)
	convert_to_csv(r'/Users/vivek/git/COMP_551_A1/Datasets/CandC_test_5.csv', test_set_5)

	# 3.2 perform linear regression (SGD)
	# create input matrix
	x_matrix = create_input_matrix(training_set_1)
	# create output array
	output_actual = create_output_vector(training_set_1)
	# print output_actual
	print x_matrix.shape
	print len(x_matrix)
	# for i in range(len(x_matrix)):
	# 	print x_matrix[i][1]

	# create random inital w_vector (123 features, 124 coefficients)
	# init_w_vector = np.random.random((124,1))
	# print x_matrix.shape
	# w_vector = np.dot(np.dot(np.linalg.inv(np.dot(x_matrix.T,x_matrix)),x_matrix.T),output_actual)
	# w_vector = np.dot(np.dot(np.linalg.inv(np.dot(x_matrix.T,x_matrix)),x_matrix.T),output_actual)

	# print w_vector.shape





