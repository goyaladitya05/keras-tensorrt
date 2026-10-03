# TensorRT backend for Keras

An inference-only [NVIDIA TensorRT](https://developer.nvidia.com/tensorrt) backend for Keras 3, built as a pluggable backend package in the same way as [keras-openvino](https://github.com/keras-team/keras-openvino) and [keras-mlx](https://github.com/keras-team/keras-mlx).

Train with any Keras backend, then run inference on NVIDIA GPUs with `KERAS_BACKEND=tensorrt`.
