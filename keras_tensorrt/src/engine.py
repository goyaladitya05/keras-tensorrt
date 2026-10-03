"""Minimal TensorRT engine helpers: build from a network, run on NumPy I/O.

TensorRT and cuda-python are imported lazily, so the package (and its
NumPy-delegated eager ops) imports on machines without a GPU.
"""

import numpy as np


def _trt():
    import tensorrt as trt

    return trt


def _cudart():
    from cuda.bindings import runtime as cudart

    return cudart


def _check(result):
    cudart = _cudart()
    err, *rest = result if isinstance(result, tuple) else (result,)
    if err != cudart.cudaError_t.cudaSuccess:
        raise RuntimeError(f"CUDA error: {err}")
    return rest[0] if len(rest) == 1 else rest


_LOGGER = None


def _logger():
    global _LOGGER
    if _LOGGER is None:
        trt = _trt()
        _LOGGER = trt.Logger(trt.Logger.WARNING)
    return _LOGGER


def build_engine(define_network, optimization_level=3, timing_cache=None):
    """Builds an engine from a network-definition callback.

    Args:
        define_network: `fn(network)` that adds inputs and layers to a
            `tensorrt.INetworkDefinition` and marks its outputs.
        optimization_level: TensorRT builder optimization level (0-5).
        timing_cache: Optional `tensorrt.ITimingCache` shared across builds.

    Returns:
        An `Engine`.
    """
    trt = _trt()
    builder = trt.Builder(_logger())
    network = builder.create_network()
    define_network(network)
    config = builder.create_builder_config()
    config.builder_optimization_level = optimization_level
    if timing_cache is not None:
        config.set_timing_cache(timing_cache, ignore_mismatch=False)
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("TensorRT engine build failed (see log above).")
    return Engine(bytes(serialized))


class Engine:
    """A deserialized TensorRT engine with fixed-shape NumPy I/O."""

    def __init__(self, serialized):
        trt = _trt()
        self._serialized = serialized
        self._engine = trt.Runtime(_logger()).deserialize_cuda_engine(
            serialized
        )
        self._context = self._engine.create_execution_context()
        names = [
            self._engine.get_tensor_name(i)
            for i in range(self._engine.num_io_tensors)
        ]
        self.input_names = [
            n
            for n in names
            if self._engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT
        ]
        self.output_names = [n for n in names if n not in self.input_names]

    def serialize(self):
        """Returns the serialized engine, e.g. to cache it on disk."""
        return self._serialized

    def __call__(self, *inputs):
        """Runs the engine synchronously on NumPy inputs.

        Device buffers are allocated per call: fine for a skeleton, not for
        a benchmark. The predict path should keep buffers and a stream alive.
        """
        trt, cudart = _trt(), _cudart()
        if len(inputs) != len(self.input_names):
            raise ValueError(
                f"Expected {len(self.input_names)} inputs, got {len(inputs)}."
            )
        ptrs, outputs = [], []
        try:
            for name, x in zip(self.input_names, inputs):
                dtype = trt.nptype(self._engine.get_tensor_dtype(name))
                x = np.ascontiguousarray(x, dtype=dtype)
                self._context.set_input_shape(name, x.shape)
                ptr = _check(cudart.cudaMalloc(max(x.nbytes, 1)))
                ptrs.append(ptr)
                _check(
                    cudart.cudaMemcpy(
                        ptr,
                        x.ctypes.data,
                        x.nbytes,
                        cudart.cudaMemcpyKind.cudaMemcpyHostToDevice,
                    )
                )
                self._context.set_tensor_address(name, ptr)
            for name in self.output_names:
                shape = tuple(self._context.get_tensor_shape(name))
                dtype = trt.nptype(self._engine.get_tensor_dtype(name))
                out = np.empty(shape, dtype=dtype)
                ptr = _check(cudart.cudaMalloc(max(out.nbytes, 1)))
                ptrs.append(ptr)
                self._context.set_tensor_address(name, ptr)
                outputs.append((out, ptr))
            stream = _check(cudart.cudaStreamCreate())
            try:
                if not self._context.execute_async_v3(stream):
                    raise RuntimeError("TensorRT execution failed.")
                _check(cudart.cudaStreamSynchronize(stream))
            finally:
                cudart.cudaStreamDestroy(stream)
            for out, ptr in outputs:
                _check(
                    cudart.cudaMemcpy(
                        out.ctypes.data,
                        ptr,
                        out.nbytes,
                        cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost,
                    )
                )
            return [out for out, _ in outputs]
        finally:
            for ptr in ptrs:
                cudart.cudaFree(ptr)
