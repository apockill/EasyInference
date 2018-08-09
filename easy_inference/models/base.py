import abc


class BaseModel(abc.ABC):
    @abc.abstractclassmethod
    def from_path(self, **kwargs):
        pass

    @abc.abstractmethod
    def predict(self, **kwargs):
        pass
