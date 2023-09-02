#function for calcualting eigen values and eigen vectors
def eig(S):
    #calculate the eigen values and eigen vectors
    eigval, eigvec = np.linalg.eig(S)

    #sorting them in decrasing order
    sortedeig  = np.argsort(-eigval)
    eigval = eigval[sortedeig]
    eigvec = eigvec[:, sortedeig]

    return (eigval, eigvec)
