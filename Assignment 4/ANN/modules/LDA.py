import numpy as np
from .stats import *

def LDA(x, y, μ, μk, K):
	d = x.shape[1]
	Sw = np.zeros((d,d))
	Sb = np.zeros((d,d))
	for c in range(μk.shape[0]):
		Sw += (x[y == c] - μk[c]).reshape(d,-1)@(x[y == c] - μk[c]).reshape(d,-1).T
		Sb += x[y == c].shape[0] * ((μk[c] - μ).reshape(-1,1)@(μk[c] - μ).reshape(-1,1).T)
	A = np.linalg.inv(Sw) @ Sb
	eigval, eigvec = np.linalg.eig(A)
	idx  = np.argsort(np.abs(eigval))[::-1]
	eigvec = eigvec[:, idx]
	return eigvec[:, :K]