import abc
import json

import easy_inference.model_loading as loading


class BaseModel(abc.ABC):
    @abc.abstractclassmethod
    def from_path(self, **kwargs):
        pass

    @abc.abstractmethod
    def predict(self, **kwargs):
        pass


class TensorflowBaseModel(BaseModel):
    @classmethod
    def from_path(cls, model_path, labels_path=None, *args, **kwargs):
        print(*args)
        model_bytes = loading.load_tf_model(model_path)
        if labels_path is not None:
            with open(labels_path, 'r') as f:
                label_str = f.read()
                labels = json.loads(label_str)
                labels = {int(key): value for key, value in labels.items()}

            return cls(model_bytes=model_bytes,
                       labels=labels,
                       *args, **kwargs)
        return cls(model_bytes=model_bytes,
                   *args, **kwargs)