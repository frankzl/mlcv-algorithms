
import numpy as np
import math

class KMeansModel:
    def __init__(self, clusters_k):
        self.k = clusters_k
        self.centroids = None
        self.cluster_assign = None

    """
    data should be a numpy array containing all the data points
    """
    def init(self, data):
        idx = np.random.randint(data.shape[0], size=self.k)
        self.centroids = data[idx,:]
        self.cluster_assign = np.full(data.shape[0], -1)

    def assign_cluster(self, data):
        original = np.array(data)
        
        current_dist = np.full(original.shape[0], math.inf)
        for idx, centroid in enumerate(self.centroids):
            shaped_cent = np.full(original.shape, centroid)

            temp_dist = np.linalg.norm(original - shaped_cent, axis=1)
            self.cluster_assign[ temp_dist < current_dist ] = idx

            current_dist = np.minimum(current_dist, temp_dist)

    def update_centroids(self, data):
        for idx in range(0, len(self.centroids)):
            idx_mask = np.where(self.cluster_assign==idx)

            relevant = data[idx_mask]
            cluster_size = len(relevant)
            self.centroids[idx] = np.sum(relevant, axis=0) / cluster_size

    def train(self, data, iterations):
        for i in range(0,iterations):
            self.assign_cluster(data)
            self.update_centroids(data)
            
    def get_clusters(self):
        return self.cluster_assign
    
    def get_centroids(self):
        return self.centroids
