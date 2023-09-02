import numpy as np

#function for calcualting top K eigen values and eigen vectors
def PCA(x, K):
    #calculate the eigen values and eigen vectors
    eigval, eigvec = np.linalg.eig(x) 
    
    #sorting them in decrasing order
    sortedeig  = np.argsort(eigval)[::-1]
    eigval = eigval[sortedeig]
    eigvec = eigvec[:, sortedeig]

    return (eigval[:K], eigvec[:, :K])