import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

from sklearn.metrics import accuracy_score
np.random.seed(0)


def load_data():
    X = np.load(r'E:\ML\projects\mnist-sample\X.npy')
    y = np.load(r'E:\ML\projects\mnist-sample\y.npy')

    # Normalize data
    X = X / 255.0

    return X, y


def softmax_batch(X):
   X = X - np.max(X,axis=1,keepdims=True)
   return np.exp(X) / np.sum(np.exp(X),axis=1,keepdims=True)


def cross_entropy_batch(y_true,y_pred):
    # - sum(y_true*log(y_pred))
    sample = -np.sum(y_true*np.log(y_pred),axis=1)
    return np.mean(sample)

def forward_batch(X_batch,w1,b1,w2,b2,w3,b3):
    # 2 hidden layers of sizes (20,15) , apply tanh to layer 1 and 2
    net1 = np.dot(X_batch,w1) + b1
    out1 = np.tanh(net1)
    net2 = np.dot(out1,w2) + b2
    out2 = np.tanh(net2)
    net3 = np.dot(out2,w3) + b3
    out3 = softmax_batch(net3)
    return out1,out2,out3

def dtanh(y):
    return 1 - y**2

def backward_batch(X_batch,y_batch,w2,w3,out1,out2,out3):
    de_dnet3 = out3 - y_batch
    de_dout2 = np.dot(de_dnet3,w3.T)  
    de_dnet2 = de_dout2 * dtanh(out2)
    de_dout1 = np.dot(de_dnet2,w2.T)
    de_dnet1 = de_dout1 * dtanh(out1)

    dw3 = np.dot(out2.T,de_dnet3)
    dw2 = np.dot(out1.T,de_dnet2)
    dw1 = np.dot(X_batch.T,de_dnet1)

    db3 = np.mean(de_dnet3,axis=0,keepdims=True)
    db2 = np.mean(de_dnet2,axis=0,keepdims=True)
    db1 = np.mean(de_dnet1,axis=0,keepdims=True)

    return dw1,db1,dw2,db2,dw3,db3

class NeuralNetworkMultiClassifier:
    def __init__(self,input_dim,hidden1_dim,hidden2_dim,output_dim):
        self.w1 = np.random.randn(input_dim,hidden1_dim)
        self.b1 = np.zeros((1,hidden1_dim))

        self.w2 = np.random.randn(hidden1_dim,hidden2_dim)
        self.b2 = np.zeros((1,hidden2_dim))

        self.w3 = np.random.randn(hidden2_dim,output_dim)
        self.b3 = np.zeros((1,output_dim))
    def train(self, X_train, y_train, X_test, y_test, learning_rate = 1e-2, n_epochs = 20, batch_size = 32):
            
            for i in range(n_epochs):
                for j in range(0,X_train.shape[0],batch_size):
                    X_batch = X_train[j : j + batch_size]
                    y_batch = y_train[j : j + batch_size]
                    out1,out2,out3 = forward_batch(X_batch,self.w1,self.b1,self.w2,self.b2,self.w3,self.b3)
                    dw1,db1,dw2,db2,dw3,db3 = backward_batch(X_batch,y_batch,self.w2,self.w3,out1,out2,out3)
                    self.w1 -= learning_rate*dw1
                    self.w2 -= learning_rate*dw2
                    self.w3 -= learning_rate*dw3
                    self.b1 -= learning_rate*db1
                    self.b2 -= learning_rate*db2
                    self.b3 -= learning_rate*db3
                __,__,out3 = forward_batch(X_test,self.w1,self.b1,self.w2,self.b2,self.w3,self.b3)

                print(f'epoch {i} , last loss = {cross_entropy_batch(y_test,out3)}')        
                y_pred = np.argmax(out3,axis=1)
                y_true = np.argmax(y_test,axis=1)
                print(f'accuaracy {accuracy_score(y_true,y_pred)}')
            

    

if __name__ == '__main__':
    X, y = load_data()
    encoder = OneHotEncoder(sparse_output=False)
    y_onehot = encoder.fit_transform(y.reshape(-1, 1))

    X_train, X_test, y_train, y_test = train_test_split(X, y_onehot, test_size=0.2, random_state=42)

    nn = NeuralNetworkMultiClassifier(X_train.shape[1], 20, 15, 10)

    nn.train(X_train, y_train, X_test, y_test)
    
