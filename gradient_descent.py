import numpy as np
from numpy.linalg import norm
import pandas as pd

def gradient_descent_linear_regression(X, t,initial_values, step_size = 0.01, precision = 0.0001, max_iter = 1000,alpha = 0.1): 
    def f_derivative(x, t, weights):
    # x: shape (n_samples, n_features)
    # t: shape (n_samples,)
    # weights: shape (n_features,)
    
        predictions = x @ weights              # shape (n_samples,)
        errors = predictions - t               # shape (n_samples,)
        gradients = (x.T @ errors) / x.shape[0]  # shape (n_features,)
        gradients[1:] += alpha * weights
        return gradients
    
    curr_state =  np.array(initial_values)
    last_state = curr_state + 400
    i = 0
    costs = []
    while norm(abs(curr_state - last_state )) > precision and i < max_iter:
        last_state = curr_state.copy()
        gradients = f_derivative(X,t,curr_state)
        curr_state -= gradients*step_size
        i+=1
    return curr_state


def costfun(x,t,weights):
    ## 1/2n (summation(1*w0 + w1X - t)^2)
    predictions = x @ weights
    error = predictions - t
    cost = np.sum(error ** 2) / (2 * x.shape[0])
    return cost

    
