import numpy as np
import statistics

def KNN(xtest, x, idx, K):
	n_tot, d = x.shape
	s = np.zeros(n_tot)
	for i in range(n_tot):
		s[i] = np.linalg.norm(xtest-x[i])
	A = idx[np.argsort(s)[:K]]
	return (statistics.mode(A), A)