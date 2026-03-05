"""
===============================================================================
MLP Regressor From Scratch
===============================================================================

Author      : SMILING07
Project     : Machine Learning Journey
File        : mlp_regressor_from_scratch.py
Description : Implementation of a Multi-Layer Perceptron (MLP) Regressor 
              built completely from scratch using NumPy.

Architecture:
    - Input Layer
    - Hidden Layer 1 (ReLU)
    - Hidden Layer 2 (ReLU)
    - Output Layer (Linear)

Loss Function:
    - Mean Squared Error (MSE)

Optimization:
    - Gradient Descent (Backpropagation implemented manually)

Features:
    - He Initialization
    - Feature Scaling using StandardScaler
    - Forward Propagation
    - Backward Propagation
    - Custom Training Loop
    - R² Score Evaluation

Dataset:
    - AQI / California Housing (Regression Task)

Purpose:
    To deeply understand neural network mathematics, gradient flow,
    weight updates, and model optimization without using high-level 
    deep learning frameworks.

Status:
    First Deep Learning Model - Built From Scratch 🚀

===============================================================================
"""
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from sklearn. model_selection import train_test_split
from sklearn.datasets import fetch_california_housing

class MLPRegressor:
    def __init__(self):
        self.W1 = None
        self.W2 = None
        self.W3 = None
        self.B1 = None
        self.B2 = None
        self.B3 = None
        self._Lambda =None
    def _MSE(self,y_real,y_pred):
        return np.mean(np.square(y_pred-y_real))+self._Lambda*(
            np.sum(np.square(self.W1))+
            np.sum(np.square(self.W2))+
            np.sum(np.square(self.W3))
        )

    def _RelU(self,Z):
        return np.maximum(0,Z)

    def _deriv_RelU(self,Z):
        return (Z > 0).astype(float)

    def _forward(self,x,weight,bias):
        return x@weight+bias

    def _backward(self,x,y,A1,A2,A3,Z1,Z2,Z3,W1,W2,W3,B1,B2,B3,lr):
        error = y-A3
        n   = y.shape[0]
        dZ3 = error
        dZ2 = dZ3@W3.T*self._deriv_RelU(Z2)
        dZ1 = dZ2@W2.T*self._deriv_RelU(Z1)
        dW3 =((-2/n)*(dZ3).T@A2).T + 2*self._Lambda*W3
        dW2 =((-2/n)*(dZ2).T@A1).T + 2*self._Lambda*W2
        dW1 =((-2/n)*(dZ1).T@x).T  + 2*self._Lambda*W1
        dB3 = (-2/n) * np.sum(dZ3)
        dB2 = (-2/n) * np.sum(dZ2)
        dB1 = (-2/n) * np.sum(dZ1)
        W1 -= dW1*lr
        W2 -= dW2*lr
        W3 -= dW3*lr
        B1 -= dB1*lr
        B2 -= dB2*lr
        B3 -= dB3*lr
        return W1,W2,W3,B1,B2,B3
    def fit(self,x ,y,lr=0.0001,epochs=100,_Lambda = 0.001)->None:
        self.W1 = np.random.randn(x.shape[1],64) *np.sqrt(2/x.shape[1])
        self.W2 = np.random.randn(64,32) * np.sqrt(2/64)
        self.W3 = np.random.randn(32,1)  * np.sqrt(2/32)
        self.B1 = np.zeros((1, 64))
        self.B2 = np.zeros((1, 32))
        self.B3 = np.zeros((1, 1))
        self._Lambda = _Lambda
        for epoch in range(epochs):
            Z1 = self._forward(x,self.W1,self.B1)
            A1 = self._RelU(Z1)
            Z2 = self._forward(A1,self.W2,self.B2)
            A2 = self._RelU(Z2)
            Z3 = self._forward(A2,self.W3,self.B3)
            A3 = (Z3)
            self.W1,self.W2,self.W3,self.B1,self.B2,self.B3 = self._backward(x,y,A1,A2,A3,Z1,Z2,Z3,self.W1,self.W2,self.W3,self.B1,self.B2,self.B3,lr)
            if (epoch%10==0):
                print(f"Epoch: {epoch} Loss: {self._MSE(y,A3)}")

        print("Training completed!")
    def predict(self,x)->np.ndarray:
        Z1 = self._forward(x,self.W1,self.B1)
        A1 = self._RelU(Z1)
        Z2 = self._forward(A1,self.W2,self.B2)
        A2 = self._RelU(Z2)
        Z3 = self._forward(A2,self.W3,self.B3)
        A3 = Z3
        return A3

# data = pd.read_csv("data/AQI.csv")
# data = data.replace([np.inf, -np.inf], np.nan) 
# data = data.dropna()  
# X = data.drop(columns=['AQI']).values
# Y = data['AQI'].values.reshape(-1,1)

#had Accuracy of R2_Score:  0.9987475375246964

data = fetch_california_housing()
X = data.data
Y = data.target
Y = Y.reshape(Y.shape[0],1)

# had Accuracy score R2_Score:  0.7519517360388108

x_train,x_test,y_train,y_test = train_test_split(
    X,Y,
    random_state=100,
    test_size=0.2
)
x_scaler  =  StandardScaler()
x_train =  x_scaler.fit_transform(x_train)
x_test  =  x_scaler.transform(x_test)

y_scaler = StandardScaler()
y_train = y_scaler.fit_transform(y_train)
y_test = y_scaler.transform(y_test)

np.random.seed(20)
model = MLPRegressor()
lr=0.1
model.fit(x_train,y_train,lr=lr,epochs=2000,_Lambda=0.0001)
y_pred = model.predict(x_test)
y_pred = y_scaler.inverse_transform(y_pred)
y_test = y_scaler.inverse_transform(y_test)
print("R2_Score: ",r2_score(y_test,y_pred))