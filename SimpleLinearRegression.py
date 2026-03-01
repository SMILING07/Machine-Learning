import numpy as np
X = [2,4,8,16]
Y = [4,8,16,32]
X = np.array(X)
Y = np.array(Y)
N = X.shape[0]
SX = np.sum(X)
SX2= np.sum(np.square(X))
SY = np.sum(Y)
SXY = np.sum(X*Y)
w   = (N*SXY -SX*SY)/(N*SX2-SX**2)
b   = (SY - np.sum(w*X))/N
print(w*int(input("Enter X: "))+b)
