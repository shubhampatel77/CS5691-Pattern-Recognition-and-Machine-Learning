import numpy as np
import os

def read_SD(path):
	train = np.genfromtxt(f'{path}/train.txt', delimiter=',', dtype=float)
	dev = np.genfromtxt(f'{path}/dev.txt', delimiter=',', dtype=float)

	X = (train[:,:2] - train[:,:2].min(axis = 0))/(train[:,:2].max(axis = 0)-train[:,:2].min(axis = 0))
	Y = train[:,2] - 1
	X_d = (dev[:,:2] - dev[:,:2].min(axis = 0))/(dev[:,:2].max(axis = 0)-dev[:,:2].min(axis = 0))
	Y_d = dev[:,2] - 1
	return (X, Y, X_d, Y_d)


def read_ID(path, dataset, type):
	curr = os.getcwd()
	path = f"{curr}/{path}/{dataset}/{type}"
			
	os.chdir(path)

	dataList = []
	files = [file for file in os.listdir()]

	for file in files:
		dataList.append(np.genfromtxt(file, delimiter=' ', dtype=float))
	os.chdir(curr)
	return np.array(dataList)


def read_digit(path, dataset, type):
	curr = os.getcwd()
	path = f"{curr}/{path}/{dataset}/{type}"
	os.chdir(path)

	dataList = []
	files = [file for file in os.listdir() if file.endswith('.mfcc')]

	for file in files:
		data = np.genfromtxt(file, delimiter=' ', skip_header=1, dtype=float)
		data = (data - data.min(axis = 0))/(data.max(axis = 0) - data.min(axis = 0))
		dataList.append(data)
	os.chdir(curr)
	datalist_new = np.array(dataList[0])
	for i in range(1,len(dataList)):
		datalist_new = np.vstack((datalist_new, np.array(dataList[i])))
	return (dataList, datalist_new)


def read_handwritting(path, dataset, type):
	curr = os.getcwd()
	path =  f"{curr}/{path}/{dataset}/{type}"
	os.chdir(path)

	dataList = []
	files = [file for file in os.listdir() if file.endswith('.txt')]

	for file in files:
		data = np.genfromtxt(file, delimiter=' ', dtype=float)[1:].reshape(-1,2)
		data = (data - data.min(axis = 0))/(data.max(axis = 0) - data.min(axis = 0))
		dataList.append(data)
	os.chdir(curr)
	datalist_new = np.array(dataList[0])
	for i in range(1,len(dataList)):
		datalist_new = np.vstack((datalist_new, np.array(dataList[i])))
	return (dataList, datalist_new)
