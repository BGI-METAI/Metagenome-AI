from abc import ABC, abstractmethod

class Embedding(ABC):
    @abstractmethod
    def get_embedding(self):
        raise NotImplementedError()

    @abstractmethod
    def get_embedding_dim(self, batch=None):
        raise NotImplementedError()

    @abstractmethod
    def to(self):
        raise NotImplementedError()