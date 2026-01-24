import numpy as np
def gaussian_pdf(x,mean,std):
    return (1 / (std * np.sqrt(2*np.pi))) * np.exp(-1/2 * ((x-mean)/std)**2)

def get_data():
    # generate data of 3 classes , 100 example , each of 5 fatures
    # all data follows gaussian
    x0 = np.random.normal(2,1,(100,5)) # represent class 0
    x1 = np.random.normal(4,1,(100,5)) # represent class 1
    x2 = np.random.normal(6,1,(100,5)) # represent class 2
    # merge all features into one data set
    X = np.vstack([x0,x1,x2])
    y = np.array([0] * 100 + [1] * 100 + [2] * 100)

    return X,y


class GNB:
    def __init__(self):
        pass

  

    def train(self,X,y):
        self.classes = np.unique(y)
        self.means , self.stds , self.proirs = {},{},{}
        for c in self.classes:
            X_c = X[y == c]
            self.means[c] = np.mean(X_c,axis=0)
            self.stds[c] = np.std(X_c,axis=0)
            self.proirs[c] = len(X_c) / len(X)

    def predict(self,X_test):
        num_samples = X_test.shape[0]
        pred = np.zeros(num_samples)
        for i in range(num_samples):
            posteriors = {}
            for c in self.classes:
                props = gaussian_pdf(X_test[i,:],self.means[c],self.stds[c])
                liklihood = np.prod(props)
                posteriors[c] = liklihood * self.proirs[c]
            pred[i] = max(posteriors,key=lambda k:posteriors[k])
        return pred
