import pytest
import numpy as np
import tensorflow as tf
from mylib.dataloader import IsingDataLoader


@pytest.fixture
def dummy_binary_data(tmp_path):
    """Fixture generates dummy .bin files for testing.
    """
    L = 8
    N = 100
    betas_list = [0.2, 0.4, 0.6, 0.8]
    samples_per_beta = N // len(betas_list)
    
    # Ensure N is divisible
    N = samples_per_beta * len(betas_list) 
    
    # Create arrays
    betas = np.repeat(betas_list, samples_per_beta).astype(np.float64)
    rng = np.random.default_rng(42)
    spins = rng.integers(0, 2, size=(N, L, L), dtype=np.int8)

    # Save to pytest's fixture temporary directory
    spins_file = tmp_path / "test_spins.bin"
    betas_file = tmp_path / "test_betas.bin"
    spins.tofile(spins_file)
    betas.tofile(betas_file)

    return spins_file, betas_file, spins, betas, L, N


def test_dataloader_recovers_exact_arrays(dummy_binary_data):
    """Test if memory mapping recovers the original arrays exactly.
    """
    spins_file, betas_file, orig_spins, orig_betas, L, N = dummy_binary_data
    
    loader = IsingDataLoader(spins_file, betas_file, L, N)
    indices = np.arange(N)
    
    loaded_spins = loader.get_spins(indices)
    loaded_betas = loader.get_betas(indices)
    
    # Use numpy testing functions for detailed errors
    np.testing.assert_array_equal(loaded_spins, orig_spins)
    np.testing.assert_array_equal(loaded_betas, orig_betas)


def test_normalize_array():
    """Test min-max normalization.
    """
    a = np.array([0.2, 0.4, 0.6])
    normalized = IsingDataLoader.normalize_array(a)
    np.testing.assert_array_almost_equal(normalized, [0.0, 0.5, 1.0])


def test_standard_training_data_shapes(dummy_binary_data):
    """Test if tf.data.Dataset batches have the correct shapes and types.
    """
    spins_file, betas_file, _, _, L, N = dummy_binary_data
    batch_size = 16
    
    loader = IsingDataLoader(spins_file, betas_file, L, N)
    train_data, val_data = loader.get_training_data(split_ratio=0.8, batch_size=batch_size, augment=False)
    
    # Take one batch
    for spins_batch, betas_batch in train_data.take(1):
        assert spins_batch.shape == (batch_size, L, L, 1)
        assert betas_batch.shape == (batch_size,)
        assert spins_batch.dtype == tf.float32
        assert betas_batch.dtype == tf.float32


def test_homogeneous_batching(dummy_binary_data):
    """Test if homogeneous batches contain only one temperature.
    """
    spins_file, betas_file, _, _, L, N = dummy_binary_data
    batch_size = 10
    
    loader = IsingDataLoader(spins_file, betas_file, L, N)
    train_data, _ = loader.get_training_data_homogeneous(split_ratio=0.8, batch_size=batch_size)
    
    for _, betas_batch in train_data.take(3):
        unique_betas = np.unique(betas_batch.numpy())
        assert len(unique_betas) == 1, f"Found mixed betas in a homogeneous batch: {unique_betas}"