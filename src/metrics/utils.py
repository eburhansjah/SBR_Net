import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

def calculate_fpr(conf_mat): # Multi-class false pos. rate from confusion matrix
    num_classes = conf_mat.shape[0]
    
    fprs = []
    for i in range(num_classes):
        TN = np.sum(conf_mat) - np.sum(conf_mat[i,:]) - np.sum(conf_mat[:,i]) + conf_mat[i,i]
        FP = np.sum(conf_mat[:,i]) - conf_mat[i,i]
        fpr = FP / (FP + TN)
        fprs.append(fpr)
        
    mean_fprs = np.mean(fprs)
    
    return mean_fprs

def calculate_metrics(y_true, y_pred):
    precision = precision_score(y_true, y_pred, average='weighted', zero_division=1)
    recall = recall_score(y_true, y_pred, average='weighted', zero_division=1)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=1)
    
    confusion_mat = confusion_matrix(y_true, y_pred)
    fpr = calculate_fpr(confusion_mat)
    
    return precision, recall, f1, fpr