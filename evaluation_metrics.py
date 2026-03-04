"""
Module: evaluation_metrics.py
Author: SMILING07
Description: 
    Custom implementation of common machine learning evaluation metrics 
    for binary classification using NumPy. Built from scratch to 
    demonstrate the mathematical foundations of model assessment.

Included Metrics:
    - Accuracy, Precision, Recall, F1-Score
    - Confusion Matrix (TP, FN, FP, TN)
"""
import numpy as np
class metrics:
    @staticmethod
    def accuracy_score(y_real,y_pred)  -> float:
        return np.mean(y_real==y_pred)
    @staticmethod
    def precision_score(y_real,y_pred) ->float:
        return np.sum((y_pred==1)&(y_pred==y_real))/np.sum(y_pred==1)
    @staticmethod
    def recall_score(y_real,y_pred)   ->float:
        return np.sum((y_pred==1)&(y_pred==y_real))/np.sum(y_real==1)
    @staticmethod
    def f1_score(y_real,y_pred)       ->float:
        prec = np.sum((y_pred==1)&(y_pred==y_real))/np.sum(y_pred==1)
        reca = np.sum((y_pred==1)&(y_pred==y_real))/np.sum(y_real==1)

        return 2*(prec*reca)/(prec+reca) if (prec+reca!=0) else 0.0
    @staticmethod
    def confusion_matrix(y_real,y_pred) -> np.ndarray:
        TP      =np.sum((y_real==1)&(y_pred==1))
        FN      =np.sum((y_real==1)&(y_pred==0))
        FP      =np.sum((y_real==0)&(y_pred==1))
        TN      =np.sum((y_real==0)&(y_pred==0))
        metrics =[[TP,FN],[FP,TN]]
        return np.array(metrics)
