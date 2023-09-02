#SVD
Y = X@np.transpose(X) #SVD is the same as EVD for X*transposeX
egval, egvec = eig(Y) #calling eig()
#initialising some variables
y =[]
n = np.arange(1,257)
for K in n:
    V = np.diag(egval[range(K)])  #taking first K eigenvalues and diagonalising
    eginv = np.linalg.inv(egvec)  #inverting eignevector matrix
    U = egvec[:, range(K)]        #taking the corresponding K eignevectors
    Uinv = eginv[range(K), :]     #taking inverted K eigenvectors
    A = np.real(U@V@Uinv)         #reconstructed matrix
    diff = Y - A
    diffT = np.transpose(diff)
    prod = diff@diffT
    norm = np.array(math.sqrt(prod.trace())) #Frobenius norm of A, B = sqrt(trace((A-B)*(A-B)')) where ' is conjugate transpose
    y = np.append(y, norm)

#plotting
plt.plot(n,y)
plt.xlabel('K')
plt.ylabel('norm')
plt.title('SVD relationship')
