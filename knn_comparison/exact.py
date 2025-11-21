import time
import numpy as np
from sklearn.neighbors import NearestNeighbors
from .base import BaseKNN

class ExactKNN(BaseKNN):
    def __init__(self, n_neighbors=20):
        super().__init__(n_neighbors)
        self.model_user = None
        self.model_item = None

    def build(self, user_embeddings, item_embeddings):

        # user index
        t0 = time.time()
        self.model_user = NearestNeighbors(
            metric='cosine', algorithm='brute',
            n_neighbors=self.n_neighbors, n_jobs=-1
        )
        self.model_user.fit(user_embeddings)
        self.build_time_user = time.time() - t0

        # film index
        t0 = time.time()
        self.model_item = NearestNeighbors(
            metric='cosine', algorithm='brute',
            n_neighbors=self.n_neighbors, n_jobs=-1
        )
        self.model_item.fit(item_embeddings)
        self.build_time_item = time.time() - t0

    def query_user(self, vector, k=20):
        t0 = time.time()
        # reshape(1, -1) - converts the vector to a 1xD matrix, because kneighbors requires a 2D input (n_samples x n_features)
        # (128,) -> (1, 128) 
        dist, idx = self.model_user.kneighbors(vector.reshape(1, -1), n_neighbors=k)
        dt = time.time() - t0
        return dist.flatten(), idx.flatten(), dt

    def query_item(self, vector, k=20):
        t0 = time.time()
        dist, idx = self.model_item.kneighbors(vector.reshape(1, -1), n_neighbors=k)
        dt = time.time() - t0
        return dist.flatten(), idx.flatten(), dt