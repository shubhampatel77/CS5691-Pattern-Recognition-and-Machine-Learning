def ROC(x, xtest, ytest, case):
    (c1, c2, c3) = trainclass(x)
    (m1, s1) = stats(c1[:, :2]); (m2, s2) = stats(c2[:, :2]); (m3, s3) = stats(c3[:, :2])
    m = np.concatenate([m1, m2, m3], axis=0)

    threshold_min = 1
    threshold_max = 0
    for i in range(len(xtest)):
        p1 = 1/3*gaussian(xtest[i,:], m1, s1)
        p2 = 1/3*gaussian(xtest[i,:], m2, s2)
        p3 = 1/3*gaussian(xtest[i,:], m3, s3)
        p = np.array([p1, p2, p3])
        if np.min(p)<=threshold_min:
            threshold_min = np.min(p)
        elif np.max(p)>=threshold_max:
            threshold_max = np.max(p)

    threshold_vec = np.linspace(threshold_min, threshold_max, 100)
    P = np.empty(3)
    TPR = []; FPR = []
    eye = np.identity(2)
    for threshold in threshold_vec:
        TP, FP = 0, 0
        TN, FN = 0, 0
        for i in range(len(xtest)):
            ypredict = ytest[i]
            for j in range(3):

                if case == 'case1':
                    cov = np.cov(x[:, :2], rowvar=False)
                elif case == 'case2':
                    s = np.dstack((s1,s2,s3)); cov = s[:, :, j]
                elif case == 'case3':
                    cov = (np.sum(np.cov(x[:, :2], rowvar=False)*eye)*0.5)*eye
                elif case == 'case4':
                    cov = np.cov(x[:, :2], rowvar=False)*eye
                elif case == 'case5':
                    s = np.dstack((s1,s2,s3)); cov = s[:, :, j]*eye

                P[j] = 1/3*gaussian(xtest[i,:], np.array([m[j, :]]), cov)

                if P[j]>=threshold:
                    if ypredict == j+1:
                        TP += 1
                    else:
                        FP += 1
                else:
                    if ypredict == j+1:
                        FN += 1
                    else:
                        TN += 1
        TPR.append(TP/(TP+FN))
        FPR.append(FP/(FP+TN))
    return(TPR, FPR)   
