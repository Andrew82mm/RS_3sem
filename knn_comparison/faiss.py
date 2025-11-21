import time
import numpy as np
import faiss
from .base import BaseKNN


class FAISSKNN(BaseKNN):
    def __init__(self, n_neighbors=20, nlist=500):
        super().__init__(n_neighbors)
        self.nlist = nlist
        self.index_user = None
        self.index_item = None

    def _build_index(self, data):
        #FAISS is highly optimized for working with float32
        data = data.astype('float32')
        faiss.normalize_L2(data)
        n, dim = data.shape
        quantizer = faiss.IndexFlatIP(dim)
        index = faiss.IndexIVFFlat(quantizer, dim, min(self.nlist, n // 10))
        index.train(data)
        index.add(data)
        index.nprobe = 20
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
        vec = vector.astype('float32').reshape(1, -1)
        faiss.normalize_L2(vec)
        dist, idx = self.index_user.search(vec, k)
        return 1 - dist.flatten(), idx.flatten(), time.time() - t0

    def query_item(self, vector, k=20):
        t0 = time.time()
        vec = vector.astype('float32').reshape(1, -1)
        faiss.normalize_L2(vec)
        dist, idx = self.index_item.search(vec, k)
        return 1 - dist.flatten(), idx.flatten(), time.time() - t0