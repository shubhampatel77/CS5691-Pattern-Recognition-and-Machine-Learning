#change the commented lines according to the chosen case
#refer to ROC for more insight
def bayesian(x, xtest):
    eye = np.identity(2)
    (c1, c2, c3) = trainclass(x)
    (m1, s1) = stats(c1[:, :2]); (m2, s2) = stats(c2[:, :2]); (m3, s3) = stats(c3[:, :2])
    #c = np.cov(x[:, :2], rowvar=False)
    #c = (np.sum(np.cov(x[:, :2], rowvar=False)*eye)*0.5)*eye
    #c = np.cov(x[:, :2], rowvar=False)*eye
    #s1=c;s2=c;s3=c
    s1 = s1*eye; s2=s2*eye; s3=s3*eye
    ytest = np.empty([0],dtype=float)
    for i in range(len(xtest)):
        gdf1 = gaussian(xtest[i,:], m1, s1)#here
        gdf2 = gaussian(xtest[i,:], m2, s2)#here
        gdf3 = gaussian(xtest[i,:], m3, s3)#here
        ytest = np.append(ytest,np.argmax([gdf1,gdf2,gdf3])+1)
    return ytest
