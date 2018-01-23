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
	
def GSD_Online(input_matrix,w_vector,output_true,alpha):
	for i in range(len(output_true)):
		w_vector[0] = w_vector[0] - (alpha)* ((np.dot(input_matrix[i],w_vector)) - output_true[i])
		

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


    # start with some w vector
    w_D2 = np.random.random((1,2))
    # print w_D2

    # linear regression, so 2 coeff
    num_coeff_D2 = 2
    # create X matrix
    x_train_D2 = create_input_matrix(input_train_D2,num_coeff_D2)
    # print x_train_D2