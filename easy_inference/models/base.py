import abc
import json

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
    def from_path(cls, model_path, labels_path=None, *args, **kwargs):
        print(*args)
        model_bytes = loading.load_tf_model(model_path)
        if labels_path is not None:
            with open(labels_path, 'r') as f:
                label_str = f.read()
<<<<<<< HEAD
                labels = json.loads(label_str)
                labels = {int(key): value for key, value in labels.items()}

            return cls(model_bytes=model_bytes,
                       labels=labels,
                       *args, **kwargs)
        return cls(model_bytes=model_bytes,
                   *args, **kwargs)
=======
            return cls(model_bytes=model_bytes, labels_unparsed=label_str)
        return cls(model_bytes=model_bytes)
>>>>>>> 483b30730ce7767c976915346e8f10c444ecef8a
