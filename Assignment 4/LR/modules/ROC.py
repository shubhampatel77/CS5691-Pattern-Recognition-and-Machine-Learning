import numpy as np

def ROC(y_true, P):
	TPR = []; FPR = []
	threshold_min, threshold_max = P.min(), P.max()
	threshold_vec = np.linspace(threshold_min - (threshold_max - threshold_min)/100, threshold_max, 100)
	for threshold in threshold_vec:
		TP, FP = 0, 0
		TN, FN = 0, 0
		for i in range(len(y_true)):
			if P[i]>threshold:
				if y_true[i] == 1:
					TP += 1
				else:
					FP += 1
			else:
				if y_true[i] == 1:
					FN += 1
				else:
					TN += 1
		TPR.append(TP/(TP+FN))
		FPR.append(FP/(FP+TN))
	return (np.array(TPR), np.array(FPR))