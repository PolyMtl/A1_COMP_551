# q2
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

	# print power
	for x in range(len(array)):
		matrix [x] = np.full((1,size_W),array[x])
		matrix [x] = np.power(matrix[x],power)

	# print np.array(matrix)
	return matrix

def calc_weight(vector):
	weight = 0.0
	for i in range(len(vector)):
		weight += (vector[i])**2

	return weight


def calc_MSE(matrix,array_true, vector, regularization_coeff):
	# calc error (sqaured)
	error = 0
	MSE = 0
	weight = float(calc_weight(vector))
	for i in range(len(array_true)):
		temp = (np.dot(matrix[i],vector) - array_true[i])**2
		error +=temp
	
	error = error/2

	# if regularization 
	if (regularization_coeff > 0):
		temp = (float(regularization_coeff)/2)* weight
		error = error + temp
	
	# 2 times error divided by number of datapoints
	MSE = 2*(error)/len(array_true)
	return MSE



# def SGD_Online(input_matrix,w_train_D2,output_train_D2,learning_rate):
# 	num_epoch = 10
# 	w_per_epoch = []
# 	for x in range(num_epoch):
# 		for i in range(len(output_train_D2)):
# 			w_train_D2[0] = w_train_D2[0] - (learning_rate) * ((np.dot(input_matrix[i].T,w_train_D2)) - output_train_D2[i])
# 			w_train_D2[1] = w_train_D2[1] - (learning_rate) * ((np.dot(input_matrix[i].T,w_train_D2)) - output_train_D2[i]) * (input_matrix[i][1])

	
# 	return np.array(w_per_epoch)

if __name__ == "__main__":
	# getting training set
	csv_path_train_D2 = "Dataset_2_train_input.csv"
	input_train_D2 = []
	with open(csv_path_train_D2, "rb") as f_obj:
		input_train_D2 = csv_reader(f_obj)

	csv_path_train_D2 = "Dataset_2_train_output.csv" 
	output_train_D2 = []
	with open(csv_path_train_D2, "rb") as f_obj:
		output_train_D2 = csv_reader(f_obj)

	# getting validation set
	csv_path_valid_D2 = "Dataset_2_valid_input.csv"
	input_valid_D2 = []
	with open(csv_path_valid_D2,"rb") as f_obj:
		input_valid_D2 = csv_reader(f_obj)
	
	csv_path_valid_D2 = "Dataset_2_valid_output.csv"
	output_valid_D2 = []
	with open(csv_path_valid_D2,"rb") as f_obj:
		output_valid_D2 = csv_reader(f_obj)


	# start with some w vector for training
	w_train_D2 = np.random.random((2,1))
	# linear regression, so 2 coeff
	num_coeff_D2 = 2
	# create X matrix
	x_train_D2 = np.array(create_input_matrix(input_train_D2,num_coeff_D2))
	x_valid_D2 = np.array(create_input_matrix(input_valid_D2,num_coeff_D2))

	learning_rate = 10**(-6)
	num_epoch = 10000
	MSE_array_train = []
	MSE_array_valid = []

	for x in range(num_epoch):
		# print x
		MSE_array_train.append(calc_MSE(x_train_D2,output_train_D2,w_train_D2,0))
		MSE_array_valid.append(calc_MSE(x_valid_D2,output_valid_D2,w_train_D2,0))
		for i in range(len(output_train_D2)):
			w_train_D2[0] = w_train_D2[0] - (learning_rate) * ((np.dot(x_train_D2[i].T,w_train_D2)) - output_train_D2[i])
			w_train_D2[1] = w_train_D2[1] - (learning_rate) * ((np.dot(x_train_D2[i].T,w_train_D2)) - output_train_D2[i]) * (x_train_D2[i][1])
	
	MSE_array_train = np.array(MSE_array_train)
	MSE_array_valid = np.array(MSE_array_valid)
	
	z = np.linspace(0,num_epoch,num_epoch)
	plt.scatter(z,MSE_array_train,color ='black', label = 'Training MSE')
	plt.scatter(z,MSE_array_valid,color ='green', label = 'validation MSE')
	plt.title('Epoch vs MSE')
	plt.xlabel('epoch')
	plt.ylabel('MSE')
	plt.legend()
	plt.show()

