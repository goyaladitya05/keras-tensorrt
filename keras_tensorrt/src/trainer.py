"""Trainer for the TensorRT backend (inference only).

`fit()` / `train_on_batch()` are not supported (SUPPORTS_GRADIENT = False).

Placeholder: `predict()` currently runs eagerly through the NumPy-delegated
ops.
"""

from keras.src.backend.numpy.trainer import Trainer as _NumpyTrainer

_INFERENCE_ONLY = (
    "The TensorRT backend is inference-only: train with the JAX, TensorFlow "
    "or PyTorch backend, then load the weights here for inference."
)


class Trainer(_NumpyTrainer):
    def fit(self, *args, **kwargs):
        raise NotImplementedError(_INFERENCE_ONLY)

    def train_on_batch(self, *args, **kwargs):
        raise NotImplementedError(_INFERENCE_ONLY)
