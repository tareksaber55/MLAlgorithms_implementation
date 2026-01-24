from collections import Counter
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsRegressor,KNeighborsClassifier


class KNN:
    def __init__(self,k,task = 'Classification'):
        self.k = k
        self.task = task
    def fit(self,X_train,y_train):
        self.X,self.y = X_train,y_train
    def predict(self,X_test):
        return np.array([self._predict(x) for x in X_test])
    def _predict(self,x):
        distances = [np.sum((x-x_train)**2) for x_train in self.X]
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_outputs = [self.y[k_indices] for i in k_indices]
        if self.task == 'Classification':
            most_common = Counter(k_nearest_outputs).most_common(1)
            return most_common[0][0]
        elif self.task == 'Regression':
            return np.mean(k_nearest_outputs)
    def is_anomaly(self,x,threshold):
        distances =  [np.sum((x-x_train)**2) for x_train in self.X]
        k_distances = np.sort(distances)[:self.k]
        return k_distances > threshold*threshold # square the sides
        
    