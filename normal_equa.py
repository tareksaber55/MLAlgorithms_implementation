import numpy as np
from numpy.linalg import norm
import pandas as pd

# represent The Analytical closed Formula of Ridge Model 
class NormalEquations():
    def __init__(self,alpha,fit_intercept = True):
           self.alpha = alpha
           self.fit_intercept = fit_intercept    
           self.weights = 0       
    def fit(self,X,y):
        if self.fit_intercept:
             X = np.hstack((np.ones((X.shape[0],1)),X))
        x_square = X.T @ X
        alpha_matrix = self.alpha * np.eye(x_square.shape[0])
        if self.fit_intercept:
            alpha_matrix[0, 0] = 0  # don’t regularize bias
        x_square += alpha_matrix
        inverse = np.linalg.inv(x_square)
        inverse_xt = inverse @ X.T
        weights = inverse_xt @ y
        self.weights = weights  
    def predict(self,x_test):
        if self.fit_intercept:
             x_test = np.hstack((np.ones((x_test.shape[0],1)),x_test))
        results = x_test @ self.weights
        return results


