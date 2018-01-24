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

# # varies regularization coeff from 0 to lamda (10 values (arbitrary))
# def vary_array(regularization_coeff, num_lamda_values):
	
# 	x = float(regularization_coeff/num_lamda_values)
# 	array = []
# 	for i in range(int(num_lamda_values)):
# 		# print x
# 		array.append(i*x)

# 	array.append(regularization_coeff)
# 	return array

# w_D1 transpose times w_D1
# def calc_weight(vector):
# 	weight = 0.0
# 	for i in range(len(vector)):
# 		weight += (vector[i])**2

# 	return weight







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

	# 20 degree poly
	num_coeff_D1 = 21 
	# regularization coeff
	regularization_coeff = 1.0


	# input matrix for train set
	x_train_D1 = np.array(create_input_matrix(input_train_D1,num_coeff_D1))

	# input matrix for valid set
	x_valid_D1 = np.array(create_input_matrix(input_valid_D1,num_coeff_D1))
	
	# if no regularization
	# calculate OPTIMAL coeff (w_D1 vector)
	# x_train_D1 transpose 
	# x_train_D1.T

	# # x_train_D1 transpose times x
	# w_D1 = np.array(np.dot(x_train_D1.T,x_train_D1))

	# # inverse of x_train_D1 transpose * x_train_D1 
	# w_D1 = np.array(np.linalg.inv(np.dot(x.T,x)))

	# # (inverse of x_train_D1 transpose x_train_D1 ) times x_train_D1 transpose 
	# w_D1= np.array(np.dot(np.linalg.inv(np.dot(x_train_D1.T,x_train_D1)),x_train_D1.T))

	# ((inverse of x_train_D1 transpose x_train_D1 ) times x_train_D1 transpose ) times output
	w_D1 = np.array(np.dot(np.dot(np.linalg.inv(np.dot(x_train_D1.T,x_train_D1)),x_train_D1.T),output_train_D1))

	# WORK ON THIS!
	# plotting
	# flipped_w_D1 = np.array(np.flipud(w_D1))
	# poly_train = np.poly1d((np.squeeze(flipped_w_D1)))
	# plt.scatter(input_train_D1,output_train_D1,color='black')
	# plt.scatter(input_valid_D1,output_valid_D1,color='blue')
	# z = np.arange(-1,1,0.01)
	# out_train = poly_train(z)
	# plt.plot(z,out_train)
	# plt.axis([-1.5, 1.5, -20, 45])
	# plt.show()

	# calc MSE
	# no regularization
	MSE_train = calc_MSE(x_train_D1,output_train_D1,w_D1,0)
	MSE_valid = calc_MSE(x_valid_D1,output_valid_D1,w_D1,0)
	# print "MSE_train: ", MSE_train
	# print "MSE_valid: ", MSE_valid


	# Q1.2
	# _______________________________ #

	# with regularization, optimal w_D1 is different
	# num_lamda_values = number of different lamdas to iterate over
	# num_lamda_values = 10
	lamda_array = np.linspace(0.000001,1,1000)
	# vary_array(regularization_coeff,num_lamda_values)
	# print lamda_array

	# calc MSE for different lamda
	MSE_train_array = []
	MSE_valid_array = []

	for i in range(len(lamda_array)):
	  w_reg = np.array(np.dot(np.dot(np.linalg.inv(np.add(np.dot(x_train_D1.T,x_train_D1), np.dot(lamda_array[i],np.eye(num_coeff_D1)))),x_train_D1.T), output_train_D1))
	  # print w_reg
	  MSE_train_array.append(calc_MSE(x_train_D1,output_train_D1,w_reg,lamda_array[i]))
	  MSE_valid_array.append(calc_MSE(x_valid_D1,output_valid_D1,w_reg,lamda_array[i]))

	
	# print "MSE_train_array: \n", MSE_train_array
	# print "MSE_valid_array: \n", MSE_valid_array


	# WORK ON THIS!


	plt.plot(lamda_array, MSE_train_array, '-g', label = 'Training_MSE')
	plt.plot(lamda_array,MSE_valid_array, '-k', label = 'Validation_MSE')
	plt.xlim(-0.1,1.1)
	plt.legend()
	plt.show()

	# picking best value of lamda
	best_MSE = min(MSE_valid_array)
	print best_MSE
	best_lamda = lamda_array[MSE_valid_array.index(best_MSE)]
	# (np.ay(MSE_valid_array).argmin())*(float(1.0/len(lamda_array)))
	print best_lamda
	w_best = np.array(np.dot(np.dot(np.linalg.inv(np.add(np.dot(x_train_D1.T,x_train_D1), np.dot(best_lamda,np.eye(num_coeff_D1)))),x_train_D1.T), output_train_D1))

	# flipped_w_best = np.array(np.flipud(w_best))
	# poly_valid = np.poly1d((np.squeeze(flipped_w_best)))
	# out_valid = poly_valid(z)
	# plt.plot(z,out_valid)
	# plt.show()

