# refactored Q2
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
		temp = (np.dot(matrix[i].T,vector) - array_true[i])**2
		error +=temp
	
	error = error/2

	# if regularization 
	if (regularization_coeff > 0):
		temp = (float(regularization_coeff)/2)* weight
		error = error + temp
	
	# 2 times error divided by number of datapoints
	MSE = 2*(error)/len(array_true)
	return MSE

def SGD_online(w_vector,input_matrix,output_true,learning_rate):
	# print "int w:	",np.array(w_vector)
	# print "init error", calc_MSE(input_matrix,output_true,w_vector,0)
	num_epoch = 10000
	w_vector_per_epoch = []
	for i in range(num_epoch):
		w_vector_per_epoch.append(np.array(w_vector))
		for j in range(len(input_matrix)):
			w_vector[0] = w_vector[0] - (learning_rate)*(np.dot(w_vector.T,input_matrix[j]) - output_true[j])
			w_vector[1] = w_vector[1] - (learning_rate)*(np.dot(w_vector.T,input_matrix[j]) - output_true[j])*input_matrix[j][1]

	w_vector_per_epoch = np.array(w_vector_per_epoch)
	return w_vector_per_epoch		

def SGD_online_2(w_vector,input_matrix,output_true,learning_rate, target_MSE):
	# print "should be same:	",np.array(w_vector)
	epoch = 0
	print "starting MSE:	", calc_MSE(input_matrix,output_true,w_vector,0)
	while True :
		# print "current mse:	", calc_MSE(input_matrix,output_true,w_vector,0)
		if(calc_MSE(input_matrix,output_true,w_vector,0)<= target_MSE):
			return epoch
		epoch = epoch + 1
		for j in range(len(input_matrix)):
			w_vector[0] = w_vector[0] - (learning_rate)*(np.dot(w_vector.T,input_matrix[j]) - output_true[j])
			w_vector[1] = w_vector[1] - (learning_rate)*(np.dot(w_vector.T,input_matrix[j]) - output_true[j])*input_matrix[j][1]




# main
if __name__ == "__main__":
	# getting training set
	csv_path_train_D2 ='/Users/vivek/git/COMP_551_A1/Datasets/Dataset_2_train_input.csv'
	input_train_D2 = []
	with open(csv_path_train_D2, "rb") as f_obj:
		input_train_D2 = csv_reader(f_obj)

	csv_path_train_D2 = '/Users/vivek/git/COMP_551_A1/Datasets/Dataset_2_train_output.csv'
	output_train_D2 = []
	with open(csv_path_train_D2, "rb") as f_obj:
		output_train_D2 = csv_reader(f_obj)

	# getting validation set
	csv_path_valid_D2 = '/Users/vivek/git/COMP_551_A1/Datasets/Dataset_2_valid_input.csv'
	input_valid_D2 = []
	with open(csv_path_valid_D2,"rb") as f_obj:
		input_valid_D2 = csv_reader(f_obj)
	
	csv_path_valid_D2 = '/Users/vivek/git/COMP_551_A1/Datasets/Dataset_2_valid_output.csv'
	output_valid_D2 = []
	with open(csv_path_valid_D2,"rb") as f_obj:
		output_valid_D2 = csv_reader(f_obj)


	# start with some w vector for training
	init_w_vector = np.random.random((2,1))
	# linear regression, so 2 coeff
	num_coeff_D2 = 2
	# create X matrix
	x_train_D2 = np.array(create_input_matrix(input_train_D2,num_coeff_D2))
	x_valid_D2 = np.array(create_input_matrix(input_valid_D2,num_coeff_D2))

	# 2.1
	learning_rate = 10**(-6)
	w_vector_per_epoch = SGD_online(init_w_vector,x_train_D2,output_train_D2,learning_rate)

	MSE_per_epoch_training = []
	MSE_per_epoch_validation = []
	# print "here:	",init_w_vector
	for i in range(len(w_vector_per_epoch)):
		MSE_per_epoch_training.append(calc_MSE(x_train_D2,output_train_D2,w_vector_per_epoch[i],0))
		MSE_per_epoch_validation.append(calc_MSE(x_valid_D2,output_valid_D2,w_vector_per_epoch[i],0))

	# print "there:	",init_w_vector
	MSE_per_epoch_training = np.array(MSE_per_epoch_training)
	MSE_per_epoch_validation = np.array(MSE_per_epoch_validation)

	lowest_MSE = min(MSE_per_epoch_validation)
	print "target_MSE:	", lowest_MSE

	# plot
	num_epoch = 10000
	z = np.linspace(0,num_epoch,num_epoch)
	plt.scatter(z,MSE_per_epoch_training,linestyle = '-', label = 'Training MSE')
	plt.scatter(z,MSE_per_epoch_validation,linestyle = '--', label = 'validation MSE')
	plt.title('Epoch vs MSE')
	plt.xlabel('epoch')
	plt.ylabel('MSE')
	plt.legend()
	plt.show()




