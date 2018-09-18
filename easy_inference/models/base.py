import abc

import easy_inference.load_utils as loading


class BaseModel(abc.ABC):
    @abc.abstractclassmethod
    def from_path(self, **kwargs):
        pass

    @abc.abstractmethod
    def predict(self, **kwargs):
        pass


class TensorflowBaseModel(BaseModel):
    @classmethod
    def from_path(cls, model_path, labels_path=None):
        model_bytes = loading.load_tf_model(model_path)

        if labels_path is not None:
            with open(labels_path, 'r') as f:
                label_str = f.read()
            return cls(model_bytes=model_bytes, labels_path=label_str)
        return cls(model_bytes=model_bytes)
