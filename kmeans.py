import numpy as np
import pandas as pd


class kmeans:
    def __init__(self,k = 3,max_iter = 300):
        self.k = k
        self.max_iter = max_iter
        self.clusters = {}
        self.centroids = []
    def intialize_centroids(self,data):
        random_indices = np.random.choice(data.shape[0],self.k,replace=False)
        self.centroids = data[random_indices]
    def dist(self,point,centriod):
        return np.sum((point-centriod)**2)
    def assign_clusters(self,data):
        self.cluster = [i for i in range(self.k)]
        for point in data:
            distances = [self.dist(point,centriod) for centriod in self.centroids]
            cluster_idx = np.argmin(distances)
            self.cluster[cluster_idx].append(point)
    def update_centroids(self):
        for cluster_idx , cluster_points in self.cluster:
            self.centroids[cluster_idx] = np.mean(cluster_points,axis=0)
    def fit(self,data):
        self.intialize_centroids(data)
        for i in range(self.max_iter):
            self.assign_clusters(data)
            prev_centroids = self.centroids
            self.update_centroids()
            if np.allclose(prev_centroids,self.centroids,rtol=1e-4):
                break
    def predict(self,data):
        predictions = []
        for point in data:
            distances = [self.dist(point,centroid) for centroid in self.centroids]
            cluster_idx = np.argmin(distances)
            predictions.append(cluster_idx)
        return predictions



    

