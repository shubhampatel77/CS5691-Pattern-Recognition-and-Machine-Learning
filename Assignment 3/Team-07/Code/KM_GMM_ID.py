from Kmeans import *
from GMM import *
from ID_Read import *
import numpy as np
import random

#Reading the training and development datasets
train_coast = read_data('coast', 'train')
dev_coast = read_data('coast', 'dev')
train_forest = read_data('forest', 'train')
dev_forest = read_data('forest', 'dev')
train_highway = read_data('highway', 'train')
dev_highway = read_data('highway', 'dev')
train_mountain = read_data('mountain', 'train')
dev_mountain = read_data('mountain', 'dev')
train_opencountry = read_data('opencountry', 'train')
dev_opencountry = read_data('opencountry', 'dev')

#Normalising data for each class
C = []

C1 = train_coast.flatten().reshape(-1, 23)
C.append((C1 - C1.min(axis = 0))/(C1.max(axis = 0)-C1.min(axis = 0)))
C2 = train_forest.flatten().reshape(-1, 23)
C.append((C2 - C2.min(axis = 0))/(C2.max(axis = 0)-C2.min(axis = 0)))
C3 = train_highway.flatten().reshape(-1, 23)
C.append((C3 - C3.min(axis = 0))/(C3.max(axis = 0)-C3.min(axis = 0)))
C4 = train_mountain.flatten().reshape(-1, 23)
C.append((C4 - C4.min(axis = 0))/(C4.max(axis = 0)-C4.min(axis = 0)))
C5 = train_opencountry.flatten().reshape(-1, 23)
C.append((C5 - C5.min(axis = 0))/(C5.max(axis = 0)-C5.min(axis = 0)))

#Mean, Covariance calculator function
def stats(x):
	μ = x.mean(0)
	Σ = np.cov(x, rowvar=False)
	return (μ, Σ)

#Initialisation for mean and covariance the same as that of given data for each cluster
def initial(x, K):
	n, d = x.shape
	μ_initial = np.zeros([K, d])
	Σ_initial = np.zeros([K, d, d])
	for k in range(K):
		index = random.randint(0, n)
		μ_initial[k,:] = x[index]
		Σ_initial[k,:,:] = stats(x)[1]
	return (μ_initial, Σ_initial)


## K Keans implementation
K = 4
μ = []
Σ = []
N = []
dk = []

for c in range(5):
	μ_initial, Σ_initial = initial(C[c], K)
	μ.append(μ_initial)
	Σ.append(Σ_initial)

μ = np.array(μ)
Σ = np.array(Σ)

for c in range(5):
	N_c, μ[c], Σ[c], dk_c = Kmeans(C[c], μ[c], K)
	N.append(N_c)
	dk.append(dk_c)

N = np.array(N)
dk = np.array(dk)

for c in range(5):
	for j in np.arange(1, 15):
		N[c], μ[c], Σ[c], dk[c] = Kmeans(C[c], μ[c], K)


##GMM implementation
π = []
dg = []
for c in range(5):
	π.append(np.zeros(K))

π = np.array(π)

for c in range(5):
	for k in range(K):	
		π[c][k] = N[c][k]/len(C[c])

for c in range(5):
	π[c], μ[c], Σ[c], dg_c = GMM(C[c], π[c], μ[c], Σ[c], K)
	dg.append(dg_c)

dg = np.array(dg)

for c in range(5):
	for j in range(1, 2):
		π[c], μ[c], Σ[c], dg[c] = GMM(C[c], π[c], μ[c], Σ[c], K)


#Predicting the class for dev data by taking argmax
prediction = np.zeros((36 * dev_coast.shape[0], 5))
score = np.zeros((36 * dev_coast.shape[0], 5))

for j in range(1):
	for i in range(36 * dev_coast.shape[0]):
		pt = dev_coast.flatten().reshape(-1, 23)[i]
		P = np.zeros(5)

		for c in range(5):
			for k in range(K):
				P[c] += π[c][k]*gaussian(pt, μ[c][k], Σ[c][k])
		prediction[i,j] = np.argmax(P)
		# score[i,j] = np.log(P.max())

## Confusion Matrix
final = np.zeros((dev_coast.shape[0], 5))
for c in range(1):
	count = 0
	for i in range((dev_coast.shape[0])):
		class_count = np.zeros(5)
		for j in range(36):
			class_count[prediction[count, c]] += 1; 
			count += 1
		
		final[i, c] = np.argmax(class_count)

CMatrix = np.zeros((5,5))
for c in range(1):
	for i in range(dev_coast.shape[0]):
		CMatrix[final[i, c], c] +=  1

print(CMatrix)