# q1
import csv
import numpy as np
import matplotlib.pyplot as plt


def csv_reader(file_obj):
	"""
	Read a csv file (input column and output column seperately)
	"""
	data = []
	reader = csv.reader(file_obj)
	for x in reader:
		data.append(float(" ".join(x))) 
	
	return data

def create_input_matrix(array, size_W):
	matrix = np.zeros((len(array),size_W))
	power = []

	for i in range(size_W):
		power.append(i)

	for x in range(len(array)):
		matrix [x] = np.full((1,size_W),array[x])
		matrix [x] = np.power(matrix[x],power)

	return matrix

def calc_MSE(matrix,array_true, vector, regularization_coeff):
	# calc error (sqaured)
	error = 0
	MSE = 0
	weight = np.dot(vector.T,vector)
	# float(calc_weight(vector))
	for i in range(len(array_true)):
		temp = (np.dot(matrix[i],vector) - array_true[i])**2
		error +=temp
	
	error = error/len(array_true)

	# if regularization 
	if (regularization_coeff > 0):
		temp = (float(regularization_coeff/2.0))* weight
		error = error + temp
	
	# 2 times error divided by number of datapoints
	
	MSE = (error)
	return MSE



	# ************************** # 
if __name__ == "__main__":

	# Q1.1
	# _______________________________ #
	# getting traning set
	csv_path_train_D1 = "Dataset_1_train_input.csv"
	input_train_D1 = []
	output_train_D1 = []
	with open(csv_path_train_D1, "rb") as f_obj:
		input_train_D1 = csv_reader(f_obj)

	csv_path_train_D1 = "Dataset_1_train_output.csv"
	with open(csv_path_train_D1, "rb") as f_obj:
		output_train_D1 = csv_reader(f_obj)

	# getting validation set
	csv_path_valid_D1 = "Dataset_1_valid_input.csv"
	input_valid_D1 = []
	output_valid_D1 = []
	with open(csv_path_valid_D1, "rb") as f_obj:
		input_valid_D1 = csv_reader(f_obj)

	csv_path_valid_D1 = "Dataset_1_valid_output.csv"
	with open(csv_path_valid_D1, "rb") as f_obj:
		output_valid_D1 = csv_reader(f_obj) 

	# getting test set
	csv_path_test_D1 = "Dataset_1_test_input.csv"
	input_test_D1 = []
	output_test_D1 = []
	with open(csv_path_test_D1, "rb") as f_obj:
		input_test_D1 = csv_reader(f_obj)

	csv_path_test_D1 = "Dataset_1_test_output.csv"
	with open(csv_path_test_D1, "rb") as f_obj:
		output_test_D1 = csv_reader(f_obj)



	# 20 degree poly
	num_coeff_D1 = 21 
	# regularization coeff
	regularization_coeff = 1.0

	# input matrix for train set
	x_train_D1 = np.array(create_input_matrix(input_train_D1,num_coeff_D1))

	# input matrix for valid set
	x_valid_D1 = np.array(create_input_matrix(input_valid_D1,num_coeff_D1))

	# input matrix for test set
	x_test_D1 = np.array(create_input_matrix(input_test_D1,num_coeff_D1))
	
	# if no regularization
	# ((inverse of x_train_D1 transpose x_train_D1 ) times x_train_D1 transpose ) times output
	w_D1 = np.array(np.dot(np.dot(np.linalg.inv(np.dot(x_train_D1.T,x_train_D1)),x_train_D1.T),output_train_D1))

	
	# ***** plotting training set ******
	# plt.scatter(input_train_D1,output_train_D1,color='black')	
	# plt.axis([-1.5, 1.5, -20, 45])
	# plt.ylabel('Output')
	# plt.xlabel('Input')
	# plt.title('Training set')
	# plt.show()

	# ***** plotting 20 degree fit on training set *****
	# plt.scatter(input_train_D1,output_train_D1,color='black', label = 'Training set')
	# flipped_w_D1 = np.array(np.flipud(w_D1)) # creating a function out of the optimal w 
	# poly_train = np.poly1d((np.squeeze(flipped_w_D1)))
	# z = np.linspace(-1,1,1000)	
	# out_train = poly_train(z)
	# plt.plot(z,out_train, label = '20 deg polynomial fit')
	# plt.axis([-1.5, 1.5, -20, 45])
	# plt.ylabel('Output')
	# plt.xlabel('Input')
	# plt.legend()
	# plt.text(0.0,20,r'MSE : $ 6.47473441332 $')
	# plt.title('20 Degree polynomial on Training set')
	# plt.show()

	# ***** plotting 20 degree fit on validation set *****
	# plt.scatter(input_valid_D1,output_valid_D1,color='black', label = 'Validation set') # plotting validation set
	# flipped_w_D1 = np.array(np.flipud(w_D1)) # creating a function out of the optimal w 
	# poly_train = np.poly1d((np.squeeze(flipped_w_D1)))
	# z = np.linspace(-1,1,1000)	
	# out_valid = poly_train(z)
	# plt.plot(z,out_valid, label = '20 deg polynomial fit')
	# plt.axis([-1.5, 1.5, -20, 45])
	# plt.ylabel('Output')
	# plt.xlabel('Input')
	# plt.legend()
	# plt.text(0.0,20,r'MSE : $ 1412.30960654 $')
	# plt.title('20 Degree polynomial on Validation set')
	# plt.show()
	

	# calc MSE
	# no regularization
	MSE_train = calc_MSE(x_train_D1,output_train_D1,w_D1,0)
	MSE_valid = calc_MSE(x_valid_D1,output_valid_D1,w_D1,0)
	# print "MSE_train: ", MSE_train
	# print "MSE_valid: ", MSE_valid


	# Q1.2
	# _______________________________ #

	# range of different lamda (reg coeff) values
	lamda_array = np.linspace(0.000001,1,1000)
	
	MSE_train_array = []
	MSE_valid_array = []
	# calc MSE for different lamda
	for i in range(len(lamda_array)):
	  # with regularization, optimal w_D1 is different
	  w_reg = np.array(np.dot(np.dot(np.linalg.inv(np.add(np.dot(x_train_D1.T,x_train_D1), np.dot(lamda_array[i],np.eye(num_coeff_D1)))),x_train_D1.T), output_train_D1))
	  MSE_train_array.append(calc_MSE(x_train_D1,output_train_D1,w_reg,lamda_array[i]))
	  MSE_valid_array.append(calc_MSE(x_valid_D1,output_valid_D1,w_reg,lamda_array[i]))

	# print "MSE_train_array: \n", MSE_train_array
	# print "MSE_valid_array: \n", MSE_valid_array


	# ***** plotting Regularization coefficient vs MSE *****
	# plt.xlabel('Regularization coefficient')
	# plt.ylabel('Mean Square Error')
	# plt.plot(lamda_array, MSE_train_array, '-g', label = 'Training_MSE')
	# plt.plot(lamda_array,MSE_valid_array, '-k', label = 'Validation_MSE')
	# plt.xlim(-0.1,1.1)
	# plt.title('Regularization coefficient vs MSE')
	# plt.legend()
	# plt.show()

	# picking best value of lamda
	best_MSE = min(MSE_valid_array)
	# print best_MSE
	best_lamda = lamda_array[MSE_valid_array.index(best_MSE)]
	# print best_lamda
	# corresponding w vector of best lamda
	w_best = np.array(np.dot(np.dot(np.linalg.inv(np.add(np.dot(x_train_D1.T,x_train_D1), np.dot(best_lamda,np.eye(num_coeff_D1)))),x_train_D1.T), output_train_D1))
	# MSE of test set with best lamda
	Test_MSE = calc_MSE(x_test_D1,output_test_D1,w_best,best_lamda)
	# print Test_MSE

