import time
import numpy as np
import hnswlib
from .base import BaseKNN

class HNSWKNN(BaseKNN):
    def __init__(self, n_neighbors=20, ef_construction=400, M=32):
        super().__init__(n_neighbors)
        self.ef_construction = ef_construction
        self.M = M
        self.index_user = None
        self.index_item = None

    def _build_index(self, data):
        dim = data.shape[1]
        index = hnswlib.Index(space='cosine', dim=dim)
        index.init_index(max_elements=data.shape[0],
                         ef_construction=self.ef_construction, M=self.M)
        index.add_items(data, np.arange(data.shape[0]))
        index.set_ef(50)
        return index

    def build(self, user_embeddings, item_embeddings):


        t0 = time.time()
        self.index_user = self._build_index(user_embeddings)
        self.build_time_user = time.time() - t0

        t0 = time.time()
        self.index_item = self._build_index(item_embeddings)
        self.build_time_item = time.time() - t0


    def query_user(self, vector, k=20):
        t0 = time.time()
        idx, dist = self.index_user.knn_query(vector, k=k)
        return dist.flatten(), idx.flatten(), time.time() - t0

    def query_item(self, vector, k=20):
        t0 = time.time()
        idx, dist = self.index_item.knn_query(vector, k=k)
        return dist.flatten(), idx.flatten(), time.time() - t0