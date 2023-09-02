import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from sklearn.metrics import confusion_matrix
import seaborn as sns

#Loading Data
train = np.loadtxt('train.txt', delimiter =',')
dev = np.loadtxt('dev.txt', delimiter =',')
realtrain = np.loadtxt('realtrain.txt', delimiter =',')
realdev = np.loadtxt('realdev.txt', delimiter =',')
nontrain = np.loadtxt('nontrain.txt', delimiter =',')
nondev = np.loadtxt('nondev.txt', delimiter =',')

#change denotes changing input data class
(c1, c2, c3) = trainclass(train) #change
(m1, s1) = stats(c1[:, :2]); (m2, s2) = stats(c2[:, :2]); (m3, s3) = stats(c3[:, :2])

#creating a grid in span(train(x1), train(x2))
min1, max1 = train[:, 0].min()-1, train[:, 0].max()+1 #change
min2, max2 = train[:, 1].min()-1, train[:, 1].max()+1 #change
x1grid = np.arange(min1, max1, .5)# change precision of grid according to the chosen data set
x2grid = np.arange(min2, max2, .5)
xx, yy = np.meshgrid(x1grid, x2grid)
r1, r2 = xx.flatten(), yy.flatten()
r1, r2 = r1.reshape((len(r1), 1)), r2.reshape((len(r2), 1))
grid = np.hstack((r1,r2)) #making pairs of x1, x2 into a column

y = bayesian(train, grid) #change
z = np.reshape(y,(((np.shape(x2grid))[0]),((np.shape(x1grid))[0])))#reshaping for contourf

#Deicision Boundary

#'Bayes with Covariance same for all classes'
#'Bayes with Covariance different for all classes'
#'Naive Bayes with C = sigma^2*I'
#'Naive Bayes with C same for all classes'
#'Naive Bayes with C different for all classes'
plt.title('Bayes with Covariance different for all classes') #change according to above 5 cases
plt.contourf(xx, yy, z, levels=2) #deicision boundary
cl1 = plt.scatter(c1[:, 0], c1[:, 1],marker="*")
cl2 = plt.scatter(c2[:, 0], c2[:, 1],marker="x")
cl3 = plt.scatter(c3[:, 0], c3[:, 1],marker=".")
plt.legend((cl1,cl2,cl3),('Class-1' , 'Class-2', 'Class-3'), loc= 'upper right')
plt.show()

#ROC Plot

print('Enter the case for which ROC is to be plotted: ', case)
case = input()
(TPR, FPR) = ROC(train, grid, y, case)
plt.plot(FPR, TPR)
plt.xlabel('FPR'); plt.ylabel('TPR'); plt.title('Bayes with Covariance different for all classes') #change according to above 5 cases
plt.show()

#GDF PLot

gdf1 = []; gdf2 = []; gdf3 = []
for i in range(len(grid)):
    g1 = gaussian(grid[i,:], m1, s1); gdf1.append(g1)
    g2 = gaussian(grid[i,:], m2, s2); gdf2.append(g2)
    g3 = gaussian(grid[i,:], m3, s3); gdf3.append(g3)


fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(111, projection='3d')

#Creating plot
ax.plot_trisurf(grid[:,0],grid[:,1],gdf3,cmap=cm.coolwarm,linewidth=0, antialiased=False)
ax.plot_trisurf(grid[:,0],grid[:,1],gdf2,cmap=cm.coolwarm,linewidth=0, antialiased=False)
ax.plot_trisurf(grid[:,0],grid[:,1],gdf1,cmap=cm.coolwarm,linewidth=0, antialiased=False)
ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.set_zlabel('output')
ax.view_init(10, 70) #change angle according to needs
plt.show()

#Confusion Matrix
y_pred = bayesian(train,dev[:,:2]) #change
y_test = dev[:,2]#change
cf_matrix = confusion_matrix(y_test, y_pred)

ax = sns.heatmap(cf_matrix, annot=True, cmap='Blues')

ax.set_title(' Non Linearly Seperable Data - Seaborn Confusion Matrix with labels\n\n');
ax.set_xlabel('\nPredicted Values')
ax.set_ylabel('Actual Values ');
plt.show()
