from abc import ABC, abstractmethod

class BaseKNN(ABC):
    """An abstract base class for all KNN implementations."""
    def __init__(self, n_neighbors: int = 20):
        self.n_neighbors = n_neighbors
        # common "counter" fields
        self.build_time_user = 0.0
        self.build_time_item = 0.0

    @abstractmethod
    def build(self, user_embeddings, item_embeddings):
        """Build indexes for users and movies."""
        raise NotImplementedError

    @abstractmethod
    def query_user(self, vector, k=20):
        """Find k nearest users."""
        raise NotImplementedError

    @abstractmethod
    def query_item(self, vector, k=20):
        """Find k nearest movies."""
        raise NotImplementedError
