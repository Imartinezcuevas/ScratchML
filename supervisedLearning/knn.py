import numpy as np
import scipy.stats

class KNN():
    def __init__(self, n_neighbors=3):
        self.n_neighbors = n_neighbors

    def euclidean_distance(self, a, b):
        return np.linalg.norm(b - a)

    def vote(self, neighbors):
        return scipy.stats.mode(neighbors)[0]

    def predict(self, X, y, New_samples):
        pred = []
        for i, sample in enumerate(New_samples):
            index = np.argsort([self.euclidean_distance(sample, x) for x in X])[:self.n_neighbors]
            k_nearest_neighbors = np.array([y[i] for i in index])
            pred.append(self.vote(k_nearest_neighbors))

        return np.array(pred)
