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

	learning_rate = 10**(-2)
	num_epoch = 100
	MSE_array_train = []
	MSE_array_valid = []

	# 2.1
	for x in range(num_epoch):
		MSE_array_train.append(calc_MSE(x_train_D2,output_train_D2,w_train_D2,0))
		MSE_array_valid.append(calc_MSE(x_valid_D2,output_valid_D2,w_train_D2,0))
		for i in range(len(output_train_D2)):
			w_train_D2[0] = w_train_D2[0] - (learning_rate) * ((np.dot(x_train_D2[i].T,w_train_D2)) - output_train_D2[i])
			w_train_D2[1] = w_train_D2[1] - (learning_rate) * ((np.dot(x_train_D2[i].T,w_train_D2)) - output_train_D2[i]) * (x_train_D2[i][1])
	
	MSE_array_train = np.array(MSE_array_train)
	MSE_array_valid = np.array(MSE_array_valid)

	# ***** plotting epoch vs MSE for training and validation ***** 
	# z = np.linspace(0,num_epoch,num_epoch)
	# plt.scatter(z,MSE_array_train,color ='black', label = 'Training MSE')
	# plt.scatter(z,MSE_array_valid,color ='green', label = 'validation MSE')
	# plt.title('Epoch vs MSE')
	# plt.xlabel('epoch')
	# plt.ylabel('MSE')
	# plt.legend()
	# plt.show()

	# 2.2
	learning_rate_array = np.linspace(1e-8, 1, 100)
	
	# b is the holds the SMALLEST validation error for each learning rate in the loop
	b = []
	for j in learning_rate_array:
		# for each learning rate, create a list that holds validation error at the end of each epoch
		c = []
		for x in range(num_epoch):
			for i in range(len(output_train_D2)):
				w_train_D2[0] = w_train_D2[0] - (j) * ((np.dot(x_train_D2[i].T,w_train_D2)) - output_train_D2[i])
				w_train_D2[1] = w_train_D2[1] - (j) * ((np.dot(x_train_D2[i].T,w_train_D2)) - output_train_D2[i]) * (x_train_D2[i][1])
			# at end of each epoch, calculate validation error and put it in list
			c.append(calc_MSE(x_valid_D2,output_valid_D2,w_train_D2,0))	
		# add to b the SMALLEST (the last) element of c
		b.append(min(c))	

	b = np.array(b)
	best_learning_rate = learning_rate_array[np.argmin(b)]


	# 2.3
	w_array = []
	for x in range(num_epoch):
		if(x == 0 or x == 5 or x == 10 or x == 15 or x == 40):
			w_array.append(np.array(w_train_D2))
		
		for i in range(len(output_train_D2)):
			w_train_D2[0] = w_train_D2[0] - (learning_rate) * ((np.dot(x_train_D2[i].T,w_train_D2)) - output_train_D2[i])
			w_train_D2[1] = w_train_D2[1] - (learning_rate) * ((np.dot(x_train_D2[i].T,w_train_D2)) - output_train_D2[i]) * (x_train_D2[i][1])

	w_array = np.array(w_array)

	# creating polynomials for 5 w vectors to plot
	flip_w1 = np.array(np.flipud(w_array[0]))
	poly_w1 = np.poly1d((np.squeeze(flip_w1)))
	print poly_w1
	flip_w2 = np.array(np.flipud(w_array[1]))
	poly_w2 = np.poly1d((np.squeeze(flip_w2)))
	print poly_w2
	flip_w3 = np.array(np.flipud(w_array[2]))
	poly_w3 = np.poly1d((np.squeeze(flip_w3)))
	print poly_w3
	flip_w4 = np.array(np.flipud(w_array[3]))
	poly_w4 = np.poly1d((np.squeeze(flip_w4)))
	print poly_w4
	flip_w5 = np.array(np.flipud(w_array[4]))
	poly_w5 = np.poly1d((np.squeeze(flip_w5)))
	print poly_w5
	
	z = np.linspace(0,2,10)
	out_poly_w1 = poly_w1(z)
	out_poly_w2 = poly_w2(z)
	out_poly_w3 = poly_w3(z)
	out_poly_w4 = poly_w4(z)
	out_poly_w5 = poly_w5(z)


	# ***** plotting 5 curves to show evolution of regression *****
	# plt.scatter(input_train_D2,output_train_D2, color = 'black', label = 'Actual')
	# plt.plot(z,out_poly_w1, color = 'green', label = 'W1: y = 0.03598 x + 0.5007')
	# plt.title('W1 on training set')
	# plt.plot(z,out_poly_w2, color = 'blue', label = 'W2: 4.113 x + 3.765')
	# plt.title('W2 on training set')
	# plt.plot(z,out_poly_w3, color = 'yellow', label = 'W3: 4.281 x + 3.626')
	# plt.title('W3 on training set')
	# plt.plot(z,out_poly_w4, color = 'orange', label = 'W4: 4.316 x + 3.597')
	# plt.title('W4 on training set')
	# plt.plot(z,out_poly_w5, color = 'pink', label = 'W5: 4.325 x + 3.589')
	# plt.title('W5 on training set')
	# plt.legend()
	# plt.xlabel('input')
	# plt.ylabel('output')
	# plt.show()





