"""`keras.ops` linalg ops for the TensorRT backend.

Placeholder: eager execution currently delegates to Keras' NumPy backend
(CPU), so every op works but nothing runs on TensorRT yet.
"""

from keras.src.backend.numpy.ops.linalg import *  # noqa: F403
