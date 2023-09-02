#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import numpy as np
import math
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.image as image
import random
from scipy.interpolate import griddata
get_ipython().run_line_magic('matplotlib', 'qt')

# In[2]:


#imdir = 'opencountry'
#img=image.imread('opencountry_cdmc935.jpg')
# print('The Shape of the image is:',img.shape)
# print('The image as array is:')
# print(img)


# In[3]:


# path = os.getcwd()
# examples = 0
# train = []
# #train = np.loadtxt(path+'/opencountry_train/'+'opencountry_land588.jpg_color_edh_entropy')
# for filename in os.listdir("opencountry_train"):
#     examples += 1
#     train.append(np.loadtxt(path+'/opencountry_train/'+filename))
#     #print(filename, examples)
# train = np.array(train).reshape(examples, 36*23)
# train
# #train = train.reshape(287, 828, 1)
# #print(np.min(train))


# In[4]:


os.chdir('/Users/shubhampatel/Desktop/PRML/Assignment 3/Synthetic Dataset')


# In[5]:


syth = np.loadtxt('train.txt', delimiter=',')
C1 = syth[syth[:,2] == 1]
C2 = syth[syth[:,2] == 2]
sythdev = np.loadtxt('dev.txt', delimiter=',')
# print(C1, C2)
# print(syth[:,2] == 1)


# In[6]:


def stats(x):
    μ = x.mean(0)
    Σ = np.cov(x, rowvar=False)
    return (μ, Σ)


# In[7]:


def gaussian(x, μ, Σ):
    d = len(μ)
    exponent = np.exp(-0.5*(x-μ)@np.linalg.inv(Σ)@(x-μ).T )
    det = 1/(np.linalg.det(Σ))**0.5
    gdf = 1/(2*np.pi)**(d/2)*det*exponent
    return gdf


# In[8]:


def initial(x, K):
    n_tot, d = x.shape
    μ_initial = np.zeros([K, d])
    Σ_initial = np.zeros([K, d, d])
    for k in np.arange(K):
        index = random.randint(0, n_tot)
        μ_initial[k, :] = x[index]
        Σ_initial[k, :, :] = stats(x)[1]
    π_initial = 1/K*np.ones(K)
    return (π_initial, μ_initial, Σ_initial)


# In[9]:


def Kmeans(x, μ, K):
    
    n_tot, d = x.shape
    err = []; N = np.zeros(K); distribution = np.zeros([n_tot, K])
    for n in range(n_tot):
        for k in range(K):
            err.append(np.linalg.norm(μ[k]-x[n]))
        err = np.array(err)
        i = np.argmin(err)
        N[i] += 1
        distribution[n, i] = 1
        err = []
    
    μ_new = np.zeros([K, d])
    for k in range(K):
        for n in range(n_tot):
            if distribution[n, k] == 1:
                μ_new[k, :] += 1/N[k]*x[n]
    
    Σ_new = np.zeros([K, d, d]); distortion = 0
    for k in range(K):
        for n in range(n_tot):
            if distribution[n, k] == 1:
                Σ_new[k, :, :] += 1/N[k]*((x[n]-μ_new[k]).reshape(d, 1)@(x[n]-μ_new[k]).reshape(1, d))
                distortion += np.linalg.norm(μ_new[k]-x[n])
            
    return (N, distribution, μ_new, Σ_new, distortion)


# In[10]:


def GMM(x, π, μ, Σ, K):
    
    n_tot, d = x.shape
    Sum = 0
    γ = np.zeros([n_tot, K])
    for n in range(n_tot):
        for k in range(K):
            Sum += π[k]*gaussian(x[n], μ[k], Σ[k])
        for k in range(K):
            γ[n, k] = π[k]*gaussian(x[n], μ[k], Σ[k])/Sum
        Sum = 0
    
    N = np.zeros(K); π_new = np.zeros(K)
    for k in range(K):
        for n in range(n_tot):
            N[k] += γ[n, k]
        π_new[k] = N[k]/n_tot
        
    μ_new = np.zeros([K, d])
    for n in range(n_tot):
        for k in range(K):
            μ_new[k, :] += γ[n, k]/N[k]*x[n]
            
    Σ_new = np.zeros([K, d, d]); distortion = 0
    for n in range(n_tot):
        for k in range(K):
            Σ_new[k, :, :] += γ[n, k]/N[k]*((x[n]-μ_new[k]).reshape(d, 1)@(x[n]-μ_new[k]).reshape(1, d))
            distortion += γ[n, k]*np.linalg.norm(μ_new[k]-x[n])
            
    return (γ, π_new, μ_new, Σ_new, distortion)      


# In[11]:


def decision(x, ytest, f):
    xx, yy = grid(x, f)[:2]
    z = ytest.reshape(xx.shape)
    return (xx, yy, z)


# In[12]:


def grid(x, f):
    min1, max1 = x[:, 0].min()-1, x[:, 0].max()+1
    min2, max2 = x[:, 1].min()-1, x[:, 1].max()+1
    x1grid = np.arange(min1, max1, f)
    x2grid = np.arange(min2, max2, f)
    xx, yy = np.meshgrid(x1grid, x2grid)
    r1, r2 = xx.flatten(), yy.flatten()
    r1, r2 = r1.reshape((len(r1), 1)), r2.reshape((len(r2), 1))
    grid = np.hstack((r1,r2))
    return (xx, yy, grid)


# In[13]:


fig = plt.figure(figsize=(10,10))
rows = 1
cols = 2
# K Means implementation
K = 16
π1_initial, μ1_initial, Σ1_initial = initial(C1[:,:2], K)
π2_initial, μ2_initial, Σ2_initial = initial(C2[:,:2], K)
for j in range(25):
    N1, dist1, μ1_new, Σ1_new, dk1 = Kmeans(C1[:,:2], μ1_initial, K)
    N2, dist2, μ2_new, Σ2_new, dk2 = Kmeans(C2[:,:2], μ2_initial, K)
    # print(np.sum(μ_new-μ_initial), np.sum(Σ_new-Σ_initial)) #np.linalg.det(Σ_initial))
    μ1_initial = μ1_new; Σ1_initial = Σ1_new
    μ2_initial = μ2_new; Σ2_initial = Σ2_new
print(dk1, dk2)
# fig.add_subplot(rows,cols,1)
# plt.scatter(C1[:,0], C1[:,1], s=1, label ='Class 1')
# plt.scatter(C2[:,0], C2[:,1], s=1, label ='Class 2')
# plt.scatter(μ1_initial[:,0], μ1_initial[:,1], s=20, c = 'red', label ='Class 1 Centroids')
# plt.scatter(μ2_initial[:,0], μ2_initial[:,1], s=20, c = 'blue', label ='Class 2 Centroids')
# plt.title('K-Means')
# plt.legend()

#GMM implementation
for j in range(5):
    γ1, π1_new, μ1_new, Σ1_new, dg1 = GMM(C1[:,:2], π1_initial, μ1_initial, Σ1_initial, K)
    γ2, π2_new, μ2_new, Σ2_new, dg2 = GMM(C2[:,:2], π2_initial, μ2_initial, Σ2_initial, K)
    #print(np.sum(μ_new-μ_initial), np.sum(Σ_new-Σ_initial))#, μ_new)#np.linalg.det(Σ_initial))
    π1_initial = π1_new; μ1_initial = μ1_new; Σ1_initial = Σ1_new
    π2_initial = π2_new; μ2_initial = μ2_new; Σ2_initial = Σ2_new
print(dg1, dg2)


# In[28]:


#1)Contour lines
xx1, yy1, g1 = grid(C1[:,:2], 0.05)
xx2, yy2, g2 = grid(C2[:,:2], 0.05)
GDF1 = np.zeros((K,len(g1)))
GDF2 = np.zeros((K,len(g2)))
for k in range(K):
    for i in range(len(g1)):
        GDF1[k, i] = gaussian(g1[i], μ1_initial[k], Σ1_initial[k])
    for i in range(len(g2)):
        GDF2[k, i] = gaussian(g2[i], μ2_initial[k], Σ2_initial[k])

GDF1[np.where(GDF1 == 0)] = math.exp(-200)
GDF2[np.where(GDF2 == 0)] = math.exp(-200)

for k in range(K):
    zz1 = GDF1[k,:].reshape(xx1.shape)
    zz2 = GDF2[k,:].reshape(xx2.shape)
    plt.contour(xx1, yy1, zz1, linewidths=0.4, cmap='Paired')
    plt.contour(xx2, yy2, zz2, linewidths=0.4, cmap='Paired')   

#2)Decision Boundary
xtest = grid(syth[:, :2], 0.05)[2]   
P1 = np.zeros(len(xtest)); P2 = np.zeros(len(xtest))
for i in range(len(xtest)):
    for k in range(K):
        P1[i] += 1/2*π1_initial[k]*gaussian(xtest[i], μ1_initial[k], Σ1_initial[k])
        P2[i] += 1/2*π2_initial[k]*gaussian(xtest[i], μ2_initial[k], Σ2_initial[k])
#P = np.vstack((P1, P2)).T
from sklearn import preprocessing
P1_norm, P2_norm = preprocessing.normalize([P1]), preprocessing.normalize([P2])
P_norm = np.vstack((P1_norm, P1_norm)).T

ytest = []; yscore =[]
for i in range(len(xtest)):
    ytest.append(np.argmax(P_norm[i,:])+1)
    yscore.append(np.max(P_norm[i,:]))
ytest = np.array(ytest); yscore = np.array(yscore)

xx, yy, z = decision(syth[:, :2], ytest, 0.05)
plt.contourf(xx, yy, z, colors = ['springgreen','lightskyblue'])

#3)GMM plots
#fig.add_subplot(rows,cols,2)
plt.scatter(C1[:,0], C1[:,1], s=1, label ='Class 1')
plt.scatter(C2[:,0], C2[:,1], s=1, label ='Class 2')
plt.scatter(μ1_initial[:,0], μ1_initial[:,1], s=20, c = 'red')
plt.scatter(μ2_initial[:,0], μ2_initial[:,1], s=20, c = 'blue')
plt.title('GMM')
plt.legend()

plt.show()


# In[26]:


#np.linspace(P.min(), P.max(),100)
from sklearn import preprocessing
P1_norm, P2_norm = preprocessing.normalize([P1]), preprocessing.normalize([P2])
P_norm = np.vstack((P1_norm, P1_norm)).T


# In[1]:


#xtest = grid(syth[:, :2], 0.1)[2]   
    #ytest.append(np.argmax(P[i,:])+1)
    #yscore.append(np.max(P[i,:]))
#ytest = np.array(ytest); yscore = np.array(yscore)
(TPR, FPR) = ROC(np.log(xtest), ytest, P_norm)
plt.plot(FPR, TPR,'r')


# In[ ]:


sythdev[499]


# In[29]:


def ROC(xtest, ytest, P):
    
    TPR = []; FPR = []
#     P1 = np.zeros(len(xtest)); P2 = np.zeros(len(xtest))
#     for i in range(len(xtest)):
#         for k in range(K):
#             P1[i] += 1/2*π1_initial[k]*gaussian(xtest[i], μ1_initial[k], Σ1_initial[k])
#             P2[i] += 1/2*π2_initial[k]*gaussian(xtest[i], μ2_initial[k], Σ2_initial[k])
#     P = np.vstack((P1, P2)).T
#     threshold_min = np.min(P)
#     threshold_max = np.max(P)
#     ytest=[]; y_predict =[]
#     for i in range(len(xtest)):
#         ytest.append(np.argmax(P[i,:])+1)
#     y_predict = ytest
     
    threshold_min, threshold_max = P.min(), P.max()
    threshold_vec = np.linspace(threshold_min, threshold_max, num=100)
    for threshold in threshold_vec:
        TP, FP = 0, 0
        TN, FN = 0, 0
        y_predict = []
#         for i in range(len(xtest[:500])):
#             if P[i, 0]>=threshold:
#                 TP += 1
#             else:
#                 FN += 1
#             if P[i, 1]>=threshold:
#                 FP += 1
#             else:
#                 TN += 1
#         for i in range(len(xtest[500:])):
#             if P[i, 1]>=threshold:
#                 TP += 1
#             else:
#                 FN += 1
#             if P[i, 0]>=threshold:
#                 FP += 1
#             else:
#                 TN += 1
            
#             if P2[i]>=threshold:
#                 y_predict.append(2)
#             else:
#                 y_predict.append(1)
#             y_predict.append(np.argmax(P[i,:])+1)
#         for i in range(len(xtest)):
#             if y_predict[i] == 1 and y_true[i] == 1:
#                 TP += 1
#             elif y_predict[i] == 1 and y_true[i] == 2:
#                 FP += 1
#             elif y_predict[i] == 2 and y_true[i] == 1:
#                 FN += 1
#             else:
#                 TN += 1
        for i in range(len(xtest)):
            for j in range(2):
                if P[i, j]>=threshold:
                    if ytest[i] == j+1:
                        TP += 1
                    else:
                        FP += 1
                else:
                    if ytest[i] == j+1:
                        FN += 1
                    else:
                        TN += 1
        TPR.append(TP/(TP+FN))
        FPR.append(FP/(FP+TN))
    return (TPR, FPR)


# In[ ]:


y_predict = []
for i in range(len(xtest)):
    if P1[i]>=.0003:
        y_predict.append(1)
    else:
        y_predict.append(2)
np.min(np.array(y_predict))


# In[ ]:





# In[ ]:


import sklearn
from sklearn.metrics import DetCurveDisplay, RocCurveDisplay
#f,t,o=sklearn.metrics.roc_curve(ytest, yscore, pos_label=1)
f,t,o=sklearn.metrics.det_curve(ytest, yscore, pos_label=1)
plt.plot(f,t)


# In[ ]:


# def plus(ds, k, random_state=42):
#     np.random.seed(random_state)
#     centroids = [ds[0]]
#     for _ in range(1, k):
#         dist_sq = np.array([min([np.inner(c-x,c-x) for c in centroids]) for x in ds])
#         probs = dist_sq/dist_sq.sum()
#         cumulative_probs = probs.cumsum()
    
#     r = np.random.rand()
#     i=0
#     for j, p in enumerate(cumulative_probs):
#         if r < p:
#             i = j
#         break
#         centroids.append(ds[i])
#     return np.array(centroids)


# In[ ]:


# from sklearn import mixture
# model = mixture.GaussianMixture(n_components=16, covariance_type='full').fit(train)
# labels = model.predict(train)
# plt.scatter(train[:, 0], train[:, 1], c=labels, s=40, cmap='viridis');

