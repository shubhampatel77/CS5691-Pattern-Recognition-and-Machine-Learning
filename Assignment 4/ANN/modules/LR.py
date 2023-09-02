import numpy as np
import pandas as pd
import math
from scipy.special import softmax

## Logistic Regression
def sigmoid(x):
	r, c = x.shape
	for row in range(r):
		for column in range(c):
			x[row, column] = 1/(1 + math.exp(-x[row, column]))
	return x

def error(x, y, w):
	z = - x @ w
	N = x.shape[0]
	error = 1/N * (np.trace(x @ w @ y.T) + np.sum(np.log(np.sum(np.exp(z), axis=1))))
	return error

def gradient(x, y, w, λ):
	z = - x @ w
	p = softmax(z, axis=1)
	N = x.shape[0]
	g = 1/N * (x.T @ (y - p)) + 2 * λ * w
	return g

def gradient_descent(x, max_iter=1500, a=0.1, λ=0.01):
	y = []
	for c in range(5):
			temp = np.zeros([len(x[c]), 5])
			for i in range(len(x[c])):
					temp[i, c] = 1
			y.append(temp)
	y = np.concatenate(y, axis=0)
	x = np.concatenate(x, axis=0)
	w = np.zeros((x.shape[1], y.shape[1]))
	step = 0
	step_lst = [] 
	error_lst = []
	w_lst = []

	while step < max_iter:
			step += 1
			w -= a * gradient(x, y, w, λ)
			step_lst.append(step); w_lst.append(w); error_lst.append(error(x, y, w))

	df = pd.DataFrame({
			'step': step_lst, 
			'error': error_lst
	})
	return df, w