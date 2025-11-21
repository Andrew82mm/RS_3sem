import time
import numpy as np
from annoy import AnnoyIndex
from .base import BaseKNN
from .utils import MemoryTracker

class AnnoyKNN(BaseKNN):
    def __init__(self, n_neighbors=20, n_trees=50):
        super().__init__(n_neighbors)
        self.n_trees = n_trees
        self.index_user = None
        self.index_item = None

    def build(self, user_embeddings, item_embeddings):
        dim = user_embeddings.shape[1]

        # user index
        t0 = time.time()
        self.index_user = AnnoyIndex(dim, 'angular')
        for i, v in enumerate(user_embeddings):
            self.index_user.add_item(i, v)
        self.index_user.build(self.n_trees)
        self.build_time_user = time.time() - t0

        # item index
        t0 = time.time()
        self.index_item = AnnoyIndex(dim, 'angular')
        # Annoy equires adding data strictly one by one in a for loop.
        for i, v in enumerate(item_embeddings):
            self.index_item.add_item(i, v)
        self.index_item.build(self.n_trees)
        self.build_time_item = time.time() - t0


    def query_user(self, vector, k=20):
        t0 = time.time()
        idx, dist = self.index_user.get_nns_by_vector(vector, k, include_distances=True)
        return np.array(dist), np.array(idx), time.time() - t0

    def query_item(self, vector, k=20):
        t0 = time.time()
        idx, dist = self.index_item.get_nns_by_vector(vector, k, include_distances=True)
        return np.array(dist), np.array(idx), time.time() - t0