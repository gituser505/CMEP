import pytest
import numpy as np
import keras as k
from keras import ops
from mylib.layers import PeriodicPadding2D, Sampling, FiLMLayer


def test_periodic_padding_shape_and_values():
    """Test if PeriodicPadding2D can correctly pad the shapes and wrap the edges for an Ising lattice.
    """
    # Create 4x4 lattice with top row = 1s and bottom = 0s
    x = np.zeros((1, 4, 4, 1), dtype=np.float32)
    x[0, 0, :, 0] = 1.0  # Top row
    
    # Using kernel_size=3 and stride=1 requires 1 pixel of padding on each side
    padding_layer = PeriodicPadding2D(kernel_size=3, strides=1)
    padded_x = padding_layer(x)
    
    # 4x4 should become 6x6
    assert padded_x.shape == (1, 6, 6, 1)
    
    # 2. Check periodic logic: The bottom padding row (index 5) should copy the top row (all 1s)
    bottom_pad_row = padded_x[0, 5, 1:5, 0]
    np.testing.assert_array_equal(bottom_pad_row, np.ones(4))


def test_sampling_determinism():
    """Test if Sampling layer is stochastic during training and deterministic during inference.
    """
    z_mean = ops.convert_to_tensor([[1.0, 2.0]], dtype="float32")
    z_log_var = ops.convert_to_tensor([[0.0, 0.0]], dtype="float32")
    
    layer = Sampling(seed=42)
    
    # Inference mode: must return exactly z_mean
    out_inference = layer([z_mean, z_log_var], training=False)
    np.testing.assert_array_equal(out_inference.numpy(), z_mean.numpy())
    
    # Training mode: must add noise (output should not equal z_mean)
    out_training = layer([z_mean, z_log_var], training=True)
    assert not np.array_equal(out_training.numpy(), z_mean.numpy())


def test_layer_serialization(tmp_path):
    """Test if custom layers can be successfully serialized and reloaded without errors.
    """
    # Build a tiny functional model utilizing the padding layer
    inputs = k.layers.Input(shape=(4, 4, 1))
    padded = PeriodicPadding2D(kernel_size=3)(inputs)
    outputs = k.layers.Conv2D(filters=2, kernel_size=3)(padded)
    model = k.Model(inputs, outputs)
    
    # Attempt save
    model_path = tmp_path/"test_model.keras"
    model.save(model_path)
    
    # Attempt load
    loaded_model = k.models.load_model(model_path)
    assert loaded_model is not None