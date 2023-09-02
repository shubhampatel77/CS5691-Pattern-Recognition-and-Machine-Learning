import numpy as np

def DTW(x, y):
	m = x.shape[0]+1
	n = y.shape[0]+1
	dtw = np.ones((m, n)) * np.inf
	dtw[0, 0] = 0
	
	for i in range(1, m):
		for j in range(1, n):
			cost = np.linalg.norm(x[i-1, :] - y[j-1, :])
			penalty = np.array([dtw[i-1, j-1], dtw[i-1, j], dtw[i, j-1]])
			dtw[i, j] = penalty.min() + cost

	return dtw