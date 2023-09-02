import  numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# importing data and taking fractions of it
def datasize(f):
    df = pd.read_csv('2d-team-7-train.txt', header=None, delimiter = ' ')
    sample = df.sample(frac=f,random_state=25)
    array = sample.to_numpy(); xone = array[:, 0]; xtwo = array[:, 1]; t = array[:,2]

    dfx = pd.read_csv('2d_team_7_dev.txt', header=None, delimiter = ' ')
    samplex = dfx.sample(frac=f,random_state=25)
    arrayx = samplex.to_numpy(); xdev = arrayx[:, 0]; ydev = arrayx[:, 1]; tdev = array[:,2]
    return (xone, xtwo, t, xdev, ydev, tdev)
xone, xtwo, t, xdev, ydev, tdev = datasize(1)
# makiing phi
def makephi(n,m):
    phi = np.ones(mama)
    done = np.arange(1,n,1)
    dtwo = np.arange(1,m,1)

    for i in done:
        phi = np.column_stack((phi,xone**i))

    for j in dtwo:
        phi = np.column_stack((phi,xtwo**j))



    return phi
# phi = np.column_stack((phi,xone))
# phi = np.column_stack((phi,xtwo))
# phi = np.column_stack((phi,xtwo**2))
# phi = np.column_stack((phi,xone**2))

phi = makephi(20,20)
#making phi for developement data
def makephid(n,m):
    phi = np.ones(mama)
    done = np.arange(1,n,1)
    dtwo = np.arange(1,m,1)

    for i in done:
        phi = np.column_stack((phi,xdev**i))

    for j in dtwo:
        phi = np.column_stack((phi,ydev**j))



    return phi
# phi = np.column_stack((phi,xone))
# phi = np.column_stack((phi,xtwo))
# phi = np.column_stack((phi,xtwo**2))
# phi = np.column_stack((phi,xone**2))

phid = makephid(20,20)


#formula to find power of x1 and x2 for the least error

rang = np.arange(2,10,1)
summ = 100
ind = [1,1]
for i in rang:
    for j in rang:
        phi = makephi(i,j)
        r = phi.T@phi
        l = np.linalg.inv(r)
        fin = l@phi.T@t
        phid = makephid(i,j)
        finald = phid@fin
        deter = (t-finald)**2
        sume = 0
        for k in deter:
            sume = sume + k
        sume
        if sume<summ:
            summ = sume
            ind = [i,j]
finald.shape


#final gives the output from model

phi = makephi(10,10)
r = phi.T@phi + 0*np.identity(19)
l = np.linalg.inv(r)
fin = l@phi.T@t
final = phi@fin
final.shape

#plots scatter of actual output vs model output

plt.scatter(t,final)
plt.title('target vs model training'); plt.xlabel('Target'); plt.ylabel('Model')
plt.show()
plt.scatter(t,finald)
plt.title('target vs model developement'); plt.xlabel('Target'); plt.ylabel('Model')
plt.show()
