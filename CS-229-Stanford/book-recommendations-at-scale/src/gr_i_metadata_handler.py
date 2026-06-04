from abc import ABC, abstractmethod

class IMetadataHandler(ABC):
    @abstractmethod
    def load(self):
        pass

    @abstractmethod
    def preprocess(self):
        pass

    @abstractmethod
    def get(self):
        pass
