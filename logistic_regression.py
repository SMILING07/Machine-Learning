"""
PROJECT: Single-Layer Perceptron (Logistic Regression) from Scratch
GOAL: Binary Classification on Heart Disease Dataset
CONCEPTS: Forward Propagation, Sigmoid Activation, Gradient Descent, Feature Scaling
ACCURACY: ~88.71%
"""
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np

class LogisticRegression:
    def __init__(self):
        self.weights = None
        self.bias  = None

    def _MSE(self,y_real,y_pred):
        return np.mean(np.square(y_real-y_pred))

    def _sigmoid(self,Z):
        return 1/(1+np.exp(-Z))

    def _derivative_sigmoid(self,Z):
        return self._sigmoid(Z)*(1-self._sigmoid(Z))

    def _forward(self,weights,x,bias):
        return x@weights +bias
    def _backward(self,y,A,Z,x,weights,bias,lr):
        error = y-A
        n  = y.shape[0]
        sig_deriv =self._derivative_sigmoid(Z)
        dw = (-2/n)*(sig_deriv*error)@x
        db = (-2/n)*np.sum(error*sig_deriv)
        weights -= dw*lr
        bias    -= db*lr
        return weights,bias

    def fit(self,x,y,epochs=1500,lr=0.1):
        self.weights = np.random.randn(x.shape[1])
        self.bias = np.random.randn()
        for epoch in range(epochs):
            Z = self._forward(self.weights,x,self.bias)
            A =self._sigmoid(Z)
            self.weights,self.bias = self._backward(y,A,Z,x,self.weights,self.bias,lr)
            if (epoch%10==0):
                print(f"Epoch: {epoch} LOSS: {self._MSE(y,A)}")
        print("Training Completed")
    def predict(self,x):
        return (self._sigmoid( x@self.weights+ self.bias)>0.5).astype(int)
np.random.seed(32)
data = pd.read_csv(r"data/heart.csv")
X    = np.array(data.T[:-1].T)
Y    = np.array(data.target)
x_train ,x_test,y_train,y_test = train_test_split(X,Y)
scaler = StandardScaler()
x_train= scaler.fit_transform(x_train)
x_test =scaler.transform(x_test)
epochs = 1500
lr = 0.1
model = LogisticRegression()
model.fit(x_train,y_train,epochs,lr)
y_pred = model.predict(x_test)
print(f"Accuracy: {accuracy_score(y_test,y_pred)}")

"""
FINAL RESULT:
Accuracy: 0.8871595330739299
NOTE: This model achieves ~88% accuracy, showing that a single-layer 
neural network can effectively classify heart disease risk from scaled features.
"""