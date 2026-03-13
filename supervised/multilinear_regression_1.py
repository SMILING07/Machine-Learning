from sklearn.metrics import r2_score
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
import numpy as np
def predict(X):
    return X.dot(weights)

data  = fetch_california_housing()
Y = data.target
X = data.data
x_train,x_test,y_train,y_test = train_test_split(
    X,Y,
    test_size=0.2,
    random_state=432
)
weights = np.linalg.inv(x_train.T.dot(x_train)).dot(x_train.T.dot(y_train))
y_pred = predict(x_test)
print("R2 Score: ",r2_score(y_test,y_pred))