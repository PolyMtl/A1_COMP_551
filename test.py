 # test
import csv
import pandas as pd

def read_csv(file_name):
	df = pd.read_csv(file_name, encoding='utf-8', header = None,comment='#', sep=',')
	y = df[1].as_matrix()
	y.shape
	return df

if __name__ == "__main__":
	df = read_csv(r'/Users/vivek/git/COMP_551_A1/Dataset_3.csv')
	df.info()
	df.columns[2:]