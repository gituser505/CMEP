import os
import pytest
import numpy as np
import tensorflow as tf
import keras as k
from keras import optimizers

from mylib.model import CVAE


@pytest.fixture
def base_hparams():
    """Provides a small set of hyperparameters for fast testing.
    """
    return {
        "L": 4,
        "latent_dim": 2,
        "enc_filters": [4, 8],
        "dec_filters": [4, 2],
        "strides": [2, 2],
        "kernels": [2, 2],
        "mlp_units": [8, 4],
        "lrlu_slope": 0.2,
        "alpha": 0.1,
        "gamma": 0.5,
        "delta": 0.5,
        "seed": 42,
        "use_physics_loss": False,
        "use_gumbel": False
    }


def test_model_fit_and_serialization(base_hparams, tmp_path):
    """Test if the model can run a training epoch and serialize/deserialize correctly.
    """
    N = 4
    hparams = base_hparams
    
    # Initialize and compile
    cvae = CVAE(hparams)
    cvae.compile(optimizer=optimizers.Adam(0.001), jit_compile=True)
    
    # Create dummy inputs
    spins = np.random.uniform(0, 1, (N, hparams["L"], hparams["L"], 1)).astype(np.float32)
    betas = np.random.uniform(0, 1, (N, 1)).astype(np.float32)
    dataset = tf.data.Dataset.from_tensor_slices((spins, betas)).batch(N)
    
    # Verify training step 
    history = cvae.fit(dataset, epochs=1, verbose=0)
    assert "total_loss" in history.history
    
    # Get predictions
    output_saved = cvae([spins, betas], training=False)
    
    # Save and reload using the tmp_path fixture
    model_path = tmp_path / "test_cvae.keras"
    cvae.save(str(model_path))
    
    loaded_cvae = k.models.load_model(str(model_path))
    output_loaded = loaded_cvae([spins, betas], training=False)
    
    # Check that saved and loaded predictions match
    for o1, o2 in zip(output_saved, output_loaded):
        np.testing.assert_array_almost_equal(o1.numpy(), o2.numpy(), decimal=4)


def test_physics_observables(base_hparams):
    """Tests the magnetization and energy functions against all up and down lattices.
    """
    hparams = base_hparams
    cvae = CVAE(hparams)
    L = cvae.L
    
    # Ground State (all spins up = 1.0)
    spins_up = k.ops.ones((1, L, L, 1), dtype="float32")
    
    # Magnetization = 1.0
    mag_up = cvae.magnetization(spins_up)
    np.testing.assert_array_almost_equal(mag_up.numpy(), [1.0])
    
    # Energy density = -2.0
    energy_up = cvae.energy(spins_up, J=1.0)
    np.testing.assert_array_almost_equal(energy_up.numpy(), [-2.0])

    # Perfect thermal state (alternating 0 and 1)
    checkerboard = np.indices((L, L)).sum(axis=0) % 2
    checkerboard = checkerboard.reshape(1, L, L, 1).astype(np.float32)
    checkerboard = k.ops.convert_to_tensor(checkerboard)
    
    # Magnetization = 0.0
    mag_check = cvae.magnetization(checkerboard)
    np.testing.assert_array_almost_equal(mag_check.numpy(), [0.0])
    
    # Energy = 2.0
    energy_check = cvae.energy(checkerboard, J=1.0)
    np.testing.assert_array_almost_equal(energy_check.numpy(), [2.0])


@pytest.mark.parametrize("stochastic_mode", [True, False])
def test_generation_outputs(base_hparams, stochastic_mode):
    """Test if generation logic outputs correct shapes, types, and binary domains.
    """
    L = base_hparams['L']
    cvae = CVAE(base_hparams)
    test_betas = k.ops.convert_to_tensor([0.2, 0.5, 0.8], dtype="float32")
    
    generated_spins = cvae.generate(test_betas, stochastic=stochastic_mode)
    
    # Assert shape: (num_betas, L, L, 1)
    assert generated_spins.shape == (3, L, L, 1)
    
    # Assert output is int8 binary
    assert generated_spins.dtype == "int8"
    unique_vals = np.unique(generated_spins.numpy())
    assert set(unique_vals).issubset({0, 1})


def test_loss_computation_conditional_paths(base_hparams):
    """Tests if physics/gumbel flags accurately shape the computed loss vector.
    """
    # Standard VAE losses only
    cvae_standard = CVAE(base_hparams)
    dummy_inputs = (k.ops.ones((2, 4, 4, 1)), k.ops.ones((2, 1)))
    dummy_outputs = [k.ops.ones((2, 4, 4, 1)), k.ops.ones((2, 2)), k.ops.ones((2, 2))]
    
    losses_standard = cvae_standard.compute_losses(dummy_inputs, dummy_outputs)
    assert len(losses_standard) == 3  # [total, recon, kl]
    
    # With physics informed losses active
    base_hparams["use_physics_loss"] = True
    cvae_physics = CVAE(base_hparams)
    losses_physics = cvae_physics.compute_losses(dummy_inputs, dummy_outputs)
    assert len(losses_physics) == 5  # [total, recon, kl, m_loss, e_loss]