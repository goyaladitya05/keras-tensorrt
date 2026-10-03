import numpy as np
import pytest

trt = pytest.importorskip("tensorrt")
cudart = pytest.importorskip("cuda.bindings.runtime")


def _has_gpu():
    err, count = cudart.cudaGetDeviceCount()
    return err == cudart.cudaError_t.cudaSuccess and count > 0


pytestmark = pytest.mark.skipif(not _has_gpu(), reason="needs an NVIDIA GPU")


def test_build_and_run_add():
    from keras_tensorrt.src.engine import build_engine

    def define(network):
        a = network.add_input("a", trt.float32, (2, 3))
        b = network.add_input("b", trt.float32, (2, 3))
        out = network.add_elementwise(a, b, trt.ElementWiseOperation.SUM)
        network.mark_output(out.get_output(0))

    engine = build_engine(define, optimization_level=0)
    x = np.arange(6, dtype=np.float32).reshape(2, 3)
    (y,) = engine(x, x)
    np.testing.assert_allclose(y, 2 * x)


def test_serialize_roundtrip():
    from keras_tensorrt.src.engine import Engine
    from keras_tensorrt.src.engine import build_engine

    def define(network):
        x = network.add_input("x", trt.float32, (4,))
        out = network.add_activation(x, trt.ActivationType.RELU)
        network.mark_output(out.get_output(0))

    engine = Engine(build_engine(define, optimization_level=0).serialize())
    (y,) = engine(np.array([-1, 0, 1, 2], dtype=np.float32))
    np.testing.assert_allclose(y, [0, 0, 1, 2])
