from sklearn.metrics import r2_score
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.preprocessing import StandardScaler,PolynomialFeatures
from sklearn.linear_model import LinearRegression

def forward(weights,x,bias):
    return x@weights+bias

def backward(y_pred,y_real,weights,bias,l2_lambda=0.01):
    error = y_real-y_pred
    n     = y_pred.shape[0]
    dw    = -(2/n)*(x_train.T @ error)+(l2_lambda*weights)
    db    = -(2/n)*np.sum(error)
    weights -= dw*lr
    bias    -= db*lr
    return weights,bias

def MSE(y_pred,y_real):
    return np.mean(np.square(y_pred-y_real))

data  = fetch_california_housing()
Y = data.target
X = data.data
poly  = PolynomialFeatures(include_bias=False,degree=2)
scaler= StandardScaler()

x_train,x_test,y_train,y_test = train_test_split(
    X,Y,
    test_size=0.2,
    random_state=432
)

x_poly_train = poly.fit_transform(x_train)  #fit then transform
x_poly_test  = poly.transform(x_test)
x_train= scaler.fit_transform(x_poly_train)
x_test = scaler.transform(x_poly_test)

# model  = LinearRegression()
# model.fit(x_train,y_train)               # had an R2_score of  0.6610240206131739

np.random.seed(432)
epochs = 500
lr     = 0.09
weights= np.random.randn(x_train.shape[1])*0.001
bias   = 0

for epoch in range(epochs):
    y_pred  = forward (weights,x_train,bias)
    weights,bias = backward(y_pred,y_train,weights,bias)
    if(epoch%10 == 0):
        print(f"Epoch: {epoch} Loss: {MSE(y_pred,y_train):4f}")

print("R2 SCORE: ",r2_score(y_test,forward(weights,x_test,bias))) #for checking r2 for skmodule use r2_score(y_test,model.predict(x_test))


"""
Feature Engineering
Using StandaredScaler PolynomialFeatures RidgeRigression
R2 SCORE:  0.6339637826758779
""" 