import pytest
import tensorflow as tf
import numpy as np

from mylib.scheduler import GumbelScheduler, PhysicalLossScheduler


class MockModel:
    """Dummy model to test callbacks without running a real training loop."""
    def __init__(self):
        self.tau = tf.Variable(1.0, dtype=tf.float32)
        self.gamma = tf.Variable(0.0, dtype=tf.float32)
        self.delta = tf.Variable(0.0, dtype=tf.float32)


@pytest.fixture
def dummy_model():
    return MockModel()


def test_gumbel_scheduler_warmup_and_decay(dummy_model):
    """Test if tau remains constant during warmup and decays correctly afterward.
    """
    schedule_params = {
        'warmup': 5,
        'tau_init': 1.0,
        'tau_min': 0.1,
        'decay_rate': 0.5
    }
    scheduler = GumbelScheduler(schedule_params)
    scheduler.set_model(dummy_model)

    # Warmup Phase (Epoch 0 to 4)
    scheduler.on_epoch_end(epoch=3)
    np.testing.assert_approx_equal(dummy_model.tau.numpy(), 1.0)
    
    scheduler.on_epoch_end(epoch=4)
    np.testing.assert_approx_equal(dummy_model.tau.numpy(), 1.0)

    # First Decay (Epoch 5 -> rel_epoch = 1, target = 0.5)
    scheduler.on_epoch_end(epoch=5)
    np.testing.assert_approx_equal(dummy_model.tau.numpy(), 0.5)

    # Second Decay (Epoch 6 -> rel_epoch = 2, target = 0.25
    scheduler.on_epoch_end(epoch=6)
    np.testing.assert_approx_equal(dummy_model.tau.numpy(), 0.25)


def test_gumbel_scheduler_min_clamp(dummy_model):
    """Test if tau never drops below its specified minimum.
    """
    schedule_params = {
        'warmup': 0,
        'tau_init': 1.0,
        'tau_min': 0.1,
        'decay_rate': 0.1
    }
    scheduler = GumbelScheduler(schedule_params)
    scheduler.set_model(dummy_model)

    # Epoch 0 -> rel_epoch = 1, target = 0.1
    scheduler.on_epoch_end(epoch=0)
    np.testing.assert_approx_equal(dummy_model.tau.numpy(), 0.1)

    # Epoch 1 -> rel_epoch = 2, target = 0.01 but should clamp at 0.1
    scheduler.on_epoch_end(epoch=1)
    np.testing.assert_approx_equal(dummy_model.tau.numpy(), 0.1)


def test_physical_loss_scheduler_warmup_and_rampup(dummy_model):
    """Test if physics loss weights respect warmup and linear ramp-up.
    """
    hparams = {'gamma': 2.0, 'delta': 4.0}
    schedule_params = {'warmup': 10, 'rampup': 4}
    
    scheduler = PhysicalLossScheduler(hparams, schedule_params)
    scheduler.set_model(dummy_model)

    # Warmup Phase (Epoch 9) -> variables should remain at their initial 0.0
    scheduler.on_epoch_begin(epoch=9)
    np.testing.assert_approx_equal(dummy_model.gamma.numpy(), 0.0)
    np.testing.assert_approx_equal(dummy_model.delta.numpy(), 0.0)

    # Rampup Step 1 (Epoch 10 -> rel_epoch = 1, progress = 0.25)
    scheduler.on_epoch_begin(epoch=10)
    np.testing.assert_approx_equal(dummy_model.gamma.numpy(), 2.0 * 0.25)
    np.testing.assert_approx_equal(dummy_model.delta.numpy(), 4.0 * 0.25)

    # Rampup Step 2 (Epoch 11 -> rel_epoch = 2, progress = 0.50)
    scheduler.on_epoch_begin(epoch=11)
    np.testing.assert_approx_equal(dummy_model.gamma.numpy(), 2.0 * 0.5)
    np.testing.assert_approx_equal(dummy_model.delta.numpy(), 4.0 * 0.5)


def test_physical_loss_scheduler_plateau(dummy_model):
    """Test if weights cap at their target values when ramp-up finishes.
    """
    hparams = {'gamma': 0.5, 'delta': 0.8}
    schedule_params = {'warmup': 5, 'rampup': 2}
    
    scheduler = PhysicalLossScheduler(hparams, schedule_params)
    scheduler.set_model(dummy_model)

    # Check Epoch 7 (Step 3/2) -> should be capped at 1.0 progress.
    scheduler.on_epoch_begin(epoch=7)
    np.testing.assert_approx_equal(dummy_model.gamma.numpy(), 0.5)
    np.testing.assert_approx_equal(dummy_model.delta.numpy(), 0.8)

    #Large future epoch test
    scheduler.on_epoch_begin(epoch=100)
    np.testing.assert_approx_equal(dummy_model.gamma.numpy(), 0.5)
    np.testing.assert_approx_equal(dummy_model.delta.numpy(), 0.8)