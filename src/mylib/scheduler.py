from keras.callbacks import Callback

class KLAnnealing(Callback):
    def __init__(self, kl_weight, warmup, rampup):
        super().__init__()
        self.kl_weight = kl_weight
        self.warmup = warmup
        self.rampup = rampup

    def on_epoch_end(self, epoch, log=None):
        if epoch < self.warmup:
            new_weight = 0.0
        else:
            new_weight = min(1.0, (epoch - self.warmup + 1) / self.rampup)
        self.kl_weight.assign(new_weight)
        print(f"KL weight = {new_weight:.3f}")


class KLAnnealingCapacity(Callback):
    def __init__(self, capacity, max_capacity, warmup, rampup):
        super().__init__()
        self.capacity = capacity
        self.max_capacity = max_capacity
        self.warmup = warmup
        self.rampup = rampup

    def on_epoch_end(self, epoch, logs=None):
        if epoch < self.warmup:
            new_capacity = 0.0
        else:
            progress = min(1.0, (epoch - self.warmup + 1) / self.rampup)
            new_capacity = progress * self.kl_max_capacity
        self.kl_capacity.assign(new_capacity)
        print(f"KL capacity (c) = {new_capacity:.3f}")


class GumbelScheduler(Callback):
    """Anneals the temperature parameter (tau) for a Gumbel-Softmax distribution.

    Exponentially decays the temperature parameter of the model's 
    Gumbel-Softmax layer at the end of each epoch, stopping at a specified minimum.

    Args:
        init_tau (float, optional): The initial temperature value. Defaults to 1.0.
        min_tau (float, optional): The minimum allowed temperature. Defaults to 0.1.
        decay_rate (float, optional): The decay rate applied per epoch. Defaults to 0.95.
    """
    def __init__(self, init_tau=1.0, min_tau=0.1, decay_rate=0.95):
        super().__init__()
        self.init_tau = init_tau
        self.min_tau = min_tau
        self.decay_rate = decay_rate

    def on_epoch_end(self, epoch, logs=None):
        """Updates the model's tau value at the end of an epoch.

        Args:
            epoch (int): The index of the current epoch.
            logs (dict, optional): Dictionary of logs from the training process. Defaults to None.
        """
        new_tau = max(self.min_tau, self.init_tau * (self.decay_rate ** (epoch + 1)))
        self.model.tau.assign(new_tau) 
        print(f" - tau annealed to: {new_tau:.4f}")


class PhysicalLossScheduler(Callback):
    """Activates physical loss components at a designated training epoch.

    Waits for a specific epoch to begin, and then assigns the 
    target weights (gamma and delta) for the physical loss functions (like 
    magnetization and energy) in the model.

    Args:
        start_epoch (int): The epoch index (0-indexed) at which to activate the physical losses.
    """
    def __init__(self, start_epoch):
        super().__init__()
        self.start_epoch = start_epoch

    def on_epoch_begin(self, epoch, logs=None): 
        """Checks the current epoch and activates physical losses if the target is reached.

        Args:
            epoch (int): The index of the current epoch.
            logs (dict, optional): Dictionary of logs from the training process. Defaults to None.
        """
        if epoch == self.start_epoch:
            self.model.gamma.assign(self.model.target_gamma)
            self.model.delta.assign(self.model.target_delta)
            print(f"\n*** Epoch {epoch + 1}: Physical losses (M & E) are now ACTIVE! ***\n")
