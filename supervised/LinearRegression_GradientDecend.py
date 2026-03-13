"""
Dataset Taken:  Moors Law 
File Name    :  moore.csv 
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

def MSE(y_pred,y_real):
    return np.mean(np.square(y_real-y_pred))

def forward(x,weight,bias):
    return x*weight+bias

def backward(y_pred,x,y_real,weight,bias):
    n = y_pred.shape[0]
    error = -(y_real-y_pred)
    dw = (2/n)*np.sum(error*x)
    db = (2/n)*np.sum(error)
    weight -= dw*lr
    bias   -= db*lr
    return weight,bias

data = pd.read_csv("supervised/data/moore.csv")
X = data.features - data.features.min()
Y = np.log(data.target)
x_train,x_test,y_train,y_test = train_test_split(
    X,
    Y,
    test_size=0.2,
    random_state=43
)

epochs = 20000
lr     = 0.0008295
np.random.seed(100)
weight = np.random.randn()*0.001
bias   = np.random.randn()*0.001

for epoch in (range(epochs)):
    y_pred = forward(x_train,weight,bias)
    weight,bias = backward(y_pred,x_train,y_train,weight,bias)
    if (epoch%10==0):
        print(f"Epoch: {epoch} Loss: {MSE(y_pred,y_train)}")

print("Weights: ",weight)
print("Bias: ",bias)
print(f"R2_score: {r2_score(y_pred=weight*x_test+bias,y_true=y_test)}")

'''
Output
Weights: 0.3453946370394579
Bias: 6.951478621560678
R2_score: 0.9773848178300789
'''