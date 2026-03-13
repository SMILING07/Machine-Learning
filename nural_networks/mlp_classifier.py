import numpy as np
from utils.evaluation_metrics import metrics

class MLPClassifier:
    def __init__(self):
        self.W1 = None
        self.W2 = None
        self.W3 = None
        self.B1 = None
        self.B2 = None
        self.B3 = None
        self.lr = None
        self.epochs  = None
        self._lambda = None

    def _forward(self,x,weight,bias) -> np.ndarray:
        return x@weight+bias
    
    def _backward(self,X,y,Z1,Z2,A1,A2,A3) -> None:
        m = y.shape[0]
        dZ3 = A3 - y
        dW3 = (A2.T@dZ3)/m
        dB3 = np.sum(dZ3,axis=0,keepdims=True)/m

        dZ2 = dZ3@self.W3.T*self._deriv_RelU(Z2)
        dW2 = (A1.T@dZ2)/m
        dB2 = np.sum(dZ2,axis=0,keepdims=True)/m

        dZ1 = dZ2@self.W2.T*self._deriv_RelU(Z1)
        dW1 = (X.T@dZ1)/m
        dB1 = np.sum(dZ1,axis=0,keepdims=True)/m

        self.W1 -= dW1 * self.lr
        self.W2 -= dW2 * self.lr
        self.W3 -= dW3 * self.lr
        self.B1 -= dB1 * self.lr
        self.B2 -= dB2 * self.lr
        self.B3 -= dB3 * self.lr

    def _cross_entropy(self,y_pred,y_real) -> int:
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        return - np.sum(y_real*np.log(y_pred))
    
    def _softmax(self,Z) -> np.ndarray:
        expZ = np.exp(Z - np.max(Z, axis=1, keepdims=True))
        return expZ / np.sum(expZ, axis=1, keepdims=True)
    
    def _RelU(self,Z) -> np.ndarray:
        return np.maximum(0,Z).astype(float)
    
    def _deriv_RelU(self ,Z) ->np.ndarray:
        return Z>0
    
    def fit(self,x_train,y_train,x_test=None,y_test=None,lr=0.01,epochs=100) -> None:
        np.random.seed(20)
        output_class_count = len(np.unique(y_train))
        y_train_real = y_train
        y_test_real = y_test
        y_train = np.eye(output_class_count)[y_train]
        y_test  = np.eye(output_class_count)[y_test]
        self.W1 = np.random.randn(x_train.shape[1],64)* np.sqrt(2/x_train.shape[1])
        self.W2 = np.random.randn(64,32)*np.sqrt(2/64)
        self.W3 = np.random.randn(32,output_class_count)*np.sqrt(2/32)
        self.B1 = np.zeros((1,64))
        self.B2 = np.zeros((1,32)) 
        self.B3 = np.zeros((1,output_class_count))
        self.lr = lr
        self.epochs = epochs
        for epoch in range(epochs):
            Z1 = self._forward(x_train,self.W1,self.B1)
            A1 = self._RelU(Z1)
            Z2 = self._forward(A1,self.W2,self.B2)
            A2 = self._RelU(Z2)
            Z3 = self._forward(A2,self.W3,self.B3)
            A3 = self._softmax(Z3)
            self._backward(x_train,y_train,Z1,Z2,A1,A2,A3)
            if(epoch%5 ==0):
                print(f"Epoch: {epoch} Loss: {self._cross_entropy(A3,y_train)}")
        print("Training_Completed! ")
        if(x_test is None or y_test is None):
            print("Data are for Training Scores: ")
            y_pred = self.predict(x_train)
            y_test = y_train_real
        else:
            print("Data are for Testing Scores")
            y_pred = self.predict(x_test)
        print("Accuracy Score: ",metrics.accuracy_score(y_test_real,y_pred))
        print("F1 Score: ",metrics.f1_score(y_test_real,y_pred))
        print("Precision Score: ",metrics.precision_score(y_test_real,y_pred))
        print("Recall Score: ",metrics.recall_score(y_test_real,y_pred))
        print("Confusion Metrics:\n",metrics.confusion_matrix(y_test_real,y_pred))

    def predict(self,x) -> np.ndarray:
        Z1 = self._forward(x,self.W1,self.B1)
        A1 = self._RelU(Z1)
        Z2 = self._forward(A1,self.W2,self.B2)
        A2 = self._RelU(Z2)
        Z3 = self._forward(A2,self.W3,self.B3)
        A3 = self._softmax(Z3)
        return np.argmax(A3,axis=1)

if __name__  == "__main__":
    import pandas as pd
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    from sklearn.datasets import load_iris
    import pickle
    iris = load_iris()
    X = np.array(iris.data)
    Y = np.array(iris.target)

    x_train,x_test,y_train,y_test = train_test_split(
        X,Y,
        test_size= 0.3,
        random_state= 43
    )
    scaler  = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test  = scaler.transform(x_test)
    model   = MLPClassifier()
    model.fit(x_train,y_train,x_test,y_test,lr=0.01,epochs=1000)
    model_data = {
        'model': model,
        'scaler': scaler
    }
    with open("nural_networks/data/mlp_classifier.pkl", 'wb') as f:
        pickle.dump(model_data, f)
    print("Model and Scaler saved!")