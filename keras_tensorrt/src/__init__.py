from keras.src.backend.common.name_scope import name_scope
from keras_tensorrt.src import ops
from keras_tensorrt.src import random
from keras_tensorrt.src import rnn
from keras_tensorrt.src.ops.core import compute_output_spec
from keras_tensorrt.src.ops.core import device_scope
from keras_tensorrt.src.variable import Variable

SUPPORTS_SPARSE_TENSORS = False
SUPPORTS_RAGGED_TENSORS = False
SUPPORTS_COMPLEX_DTYPES = False
SUPPORTS_GRADIENT = False  # inference-only backend
IS_THREAD_SAFE = True

distribution_lib = None
