"""Variables for the TensorRT backend.

Placeholder: a NumPy-backed variable (from Keras' NumPy backend). When ops
move to the lazy graph, variable values become graph constants / engine
weights; TensorRT's refit API can update built engines after `load_weights()`.
"""

from keras.src.backend.numpy.ops.core import Variable as _NumpyVariable


class Variable(_NumpyVariable):
    pass
