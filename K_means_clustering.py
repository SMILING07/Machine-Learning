"""

# K-Means Clustering From Scratch

Author      : SMILING07
Project     : Machine Learning Journey
File        : kmeans_customer_segmentation.py
Description : Implementation of the K-Means clustering algorithm built
completely from scratch using NumPy.

Algorithm Overview:
- Random centroid initialization
- Iterative cluster assignment using Euclidean distance
- Centroid recomputation based on cluster means
- Convergence check when centroids stop changing

Features Implemented:
- Custom K-Means class
- Centroid initialization using random sampling
- Distance calculation using squared Euclidean distance
- Iterative centroid updates
- Within Cluster Sum of Squares (WCSS) calculation
- Elbow Method for optimal K selection
- Cluster labeling and dataset segmentation
- Visualization of elbow curve using Matplotlib

Dataset:
- Mall Customers Dataset
(Customer demographic and spending behavior data)

Dataset Features Used:
- Gender (encoded)
- Age
- Annual Income (k$)
- Spending Score (1-100)

Goal:
Perform customer segmentation by grouping customers with
similar purchasing behavior and income characteristics.

Output:
- Optimal number of clusters using Elbow Method
- Cluster labels assigned to each customer
- Segmented dataset exported to CSV

Dependencies:
- NumPy
- Pandas
- Matplotlib

Notes:
This implementation is educational and designed to demonstrate
the internal mechanics of the K-Means clustering algorithm
without relying on high-level machine learning libraries.

Status:
Unsupervised Learning Model — Built From Scratch 🚀
===================================================

"""

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.datasets import load_iris

class KMeansCluster:
    def __init__(self):
        self.centroids = None
        self.k=None

    def fit(self,x,k=3):
        idx    = np.random.choice(x.shape[0],size=k,replace=False)
        self.centroids  = x[idx]
        self.k = k
        old_centroids   = np.zeros(shape=(k,self.centroids.shape[1]))
        while not (old_centroids==self.centroids).all():
            prediction    = self.predict(x)
            centroid_new  = np.zeros(shape=(k,self.centroids.shape[1]))
            size          = np.zeros(shape=(self.k))
            for i in range(prediction.shape[0]):
                centroid_new[prediction[i]] = centroid_new[prediction[i]]+x[i]
                size[prediction[i]]         = size[prediction[i]]+1
            old_centroids  = self.centroids
            self.centroids = centroid_new/(size[:, np.newaxis] + 1e-10)
            
    def predict(self,x):
        prediction = []
        for i in range(x.shape[0]):
            dist   = []
            for j in range(self.k):
                dist.append(np.sum(np.square(self.centroids[j]-x[i])))
            prediction.append(np.argmin(dist))
        prediction = np.array(prediction).reshape(-1,1)
        return prediction
    
    def error(self, x):
        labels     = self.predict(x).flatten()
        total_wcss = 0
        for i in range(self.k):
            cluster_points = x[labels == i]
            if len(cluster_points) > 0:
                cluster_error = np.sum(np.square(cluster_points - self.centroids[i]))
                total_wcss += cluster_error
                print(f"Cluster {i} Internal Error: {cluster_error:.2f}")
        print(f"Total WCSS (Inertia): {total_wcss:.2f}")
        return total_wcss
    
np.random.seed(20)
data           = pd.read_csv(r"/home/smiling/Documents/ML/Machine-Learning/data/Mall_Customers.csv")
data           = data.drop("CustomerID", axis=1)
data["Gender"] = data["Gender"].replace({"Male":0,"Female":1})
# data = load_iris()
# data=data.data

x     = np.array(data)
y     = []
model = KMeansCluster()
max_k = 50
for i in range(1,max_k):
    K = i
    model.fit(x,K)
    model.predict(x).flatten()
    y.append(model.error(x))
k     = np.arange(1,max_k)
plt.plot(k, y, 'ro-') 
plt.title('The Elbow Method for Optimal K')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('WCSS (Error)')
plt.grid(True)
plt.show()
#by the graph the optimal K value is 6 by Elbow method
K    = 5
model.fit(x,K)
data["customer_labels"] = model.predict(x)
data.to_csv(r"data/Segmented_customer.csv")