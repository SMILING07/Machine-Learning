import pandas as pd
from sklearn.model_selection import train_test_split
from evaluation_metrics import metrics
import numpy as np
from sklearn.preprocessing import StandardScaler

class KNearestNeighbour:
    def __init__(self):
        self.X =None
        self.Y = None
        self.K = None
    def fit(self,X,Y,K):
        self.X = X 
        self.Y = Y
        self.K = K
    def predict(self,x_test):
        y_pred=[]
        for i in range(x_test.shape[0]):
            dist =[]
            for j in range(self.X.shape[0]):
                d = np.sqrt(np.sum(np.square(self.X[j]-x_test[i])))
                dist.append((d,self.Y[j]))
            dist.sort(key = lambda x:x[0])
            val = [dist[m][1] for m in range(self.K)]
            prediction = max(set(val), key=val.count)
            y_pred.append(prediction)
        return np.array(y_pred)


np.random.seed(2311)
data = pd.read_csv(r"data/heart.csv")
Y = np.array(data.target)
X = np.array(data.T[:13].T)
x_train,x_test,y_train,y_test =  train_test_split(
    X,Y,
    random_state=43,
    test_size= 0.2
)
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

model = KNearestNeighbour()
model.fit(x_train,y_train,K=3)
y_pred  = model.predict(x_test)

print("Accuracy Score : ",metrics.accuracy_score(y_test,y_pred))
print("Precision Score : ",metrics.precision_score(y_test,y_pred))
print("Recall Score : ",metrics.recall_score(y_test,y_pred))
print("F1 Score : ",metrics.f1_score(y_test,y_pred))
print("Confusion metrics: \n",metrics.confusion_matrix(y_test,y_pred))