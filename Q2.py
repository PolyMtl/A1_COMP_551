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

	for x in range(len(array)):
		matrix [x] = np.full((1,size_W),array[x])
		matrix [x] = np.power(matrix[x],power)

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



def GSD_Online(input_matrix,w_vector,output_true,alpha):
	MSE = []
	num_epoch = 10000
	for x in range(num_epoch):
		for i in range(len(output_true)):
			w_vector[0] = w_vector[0] - (alpha) * ((np.dot(input_matrix[i].T,w_vector)) - output_true[i])
			w_vector[1] = w_vector[1] - (alpha) * ((np.dot(input_matrix[i].T,w_vector)) - output_true[i]) * (input_matrix[i][1])
	
		# print calc_MSE(input_matrix,output_true,w_vector,0)
		MSE.append(calc_MSE(input_matrix,output_true,w_vector,0))
	
	# print w_vector
	# print len(MSE)
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


    csv_path_valid_D2 = "Dataset_2_valid_input.csv"
    input_valid_D2 = []
    with open(csv_path_valid_D2,"rb") as f_obj:
    	input_valid_D2 = csv_reader(f_obj)
    
    csv_path_valid_D2 = "Dataset_2_valid_output.csv"
    output_valid_D2 = []
    with open(csv_path_valid_D2,"rb") as f_obj:
    	output_valid_D2 = csv_reader(f_obj)

    


    # start with some w vector
    w_D2 = np.random.random((2,1))
    # print w_D2

    # linear regression, so 2 coeff
    num_coeff_D2 = 2
    # create X matrix
    x_train_D2 = np.array(create_input_matrix(input_train_D2,num_coeff_D2))
    x_valid_D2 = np.array(create_input_matrix())

    MSE_array_train = np.array(GSD_Online(x_train_D2,w_D2,output_train_D2,10**(-6)))
    MSE_array_valid = np.array(GSD_Online())

    # plot
    # print MSE_array_train

    z = np.arange(0,10000,1)
    plt.scatter(z,MSE_array_train,color ='black')

    plt.show()

