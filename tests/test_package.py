# Import keras first, as in real use: keras imports keras_tensorrt.src when
# KERAS_BACKEND=tensorrt, so importing the backend module directly first
# would be circular.
import keras  # noqa: F401
import keras_tensorrt
from keras_tensorrt import src


def test_version():
    assert keras_tensorrt.__version__


def test_backend_flags():
    # Inference-only backend.
    assert src.SUPPORTS_GRADIENT is False
    assert src.Variable is not None
    assert src.name_scope is not None


def test_fit_is_rejected():
    import numpy as np
    import pytest

    if keras.backend.backend() != "tensorrt":
        pytest.skip("needs KERAS_BACKEND=tensorrt")
    model = keras.Sequential([keras.Input((4,)), keras.layers.Dense(2)])
    model.compile(optimizer="sgd", loss="mse")
    with pytest.raises(NotImplementedError, match="inference-only"):
        model.fit(np.ones((2, 4)), np.ones((2, 2)), verbose=0)


def test_predict_runs():
    import numpy as np
    import pytest

    if keras.backend.backend() != "tensorrt":
        pytest.skip("needs KERAS_BACKEND=tensorrt")
    model = keras.Sequential([keras.Input((4,)), keras.layers.Dense(2)])
    assert model.predict(np.ones((3, 4)), verbose=0).shape == (3, 2)
