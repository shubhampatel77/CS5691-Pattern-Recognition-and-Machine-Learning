#EVD
egval, egvec = eig(X) #calling eig()
#initialising some variables
x =[]
n = np.arange(1,257)
for K in n:
    V = np.diag(egval[range(K)])  #taking first K eigenvalues and diagonalising
    eginv = np.linalg.inv(egvec)  #inverting eignevector matrix
    U = egvec[:, range(K)]        #taking the corresponding K eignevectors
    Uinv = eginv[range(K), :]     #taking inverted K eigenvectors
    A = np.real(U@V@Uinv)         #reconstructed matrix
    diff = X - A
    diffT = np.transpose(diff)
    prod = diff@diffT
    norm = np.array(math.sqrt(prod.trace())) #Frobenius norm of A, B = sqrt(trace((A-B)*(A-B)')) where ' is conjugate transpose
    x = np.append(x, norm)

#shows how the reconstructed and error images vary  with t
N = np.array([10, 50, 100, 150, 200, 250, 255])
for t in N:
    V = np.diag(egval[range(t)])
    eginv = np.linalg.inv(egvec)
    U = egvec[:, range(t)]
    Uinv = eginv[range(t), :]
    A = np.real(U@V@Uinv)
    re = Image.fromarray(A) #reconstructed image
    re = re.convert("L")
    er = Image.fromarray(X-A) #error image
    er = er.convert("L")
    re.show(), er.show()

#plotting
plt.plot(n,x)
plt.xlabel('K')
plt.ylabel('norm')
plt.title('EVD relationship')
