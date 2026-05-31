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


class PhysicsScheduler(Callback):
    def __init__(self, hparams, schedule_params):
        super().__init__()
        self.gamma = hparams['gamma']
        self.delta = hparams['delta']
        self.warmup = schedule_params['warmup']
        self.rampup = schedule_params['rampup']

    def on_epoch_end(self, epoch, logs=None):
        if epoch < self.warmup:
            return
        rel_epoch = epoch - self.warmup + 1
        ramp_progress = min(1.0, rel_epoch / float(self.rampup))
        self.model.gamma.assign(self.gamma * ramp_progress)
        self.model.delta.assign(self.delta * ramp_progress)
        print(f"gamma={self.model.gamma.numpy():.4f}, delta={self.model.delta.numpy():.4f}")


class GumbelScheduler(Callback):
    def __init__(self, schedule_params):
        super().__init__()
        self.warmup = schedule_params['warmup']
        self.tau_init = schedule_params['tau_init']
        self.tau_min = schedule_params['tau_min']
        self.decay_rate = schedule_params['decay_rate']

    def on_epoch_end(self, epoch, logs=None):
        if epoch < self.warmup:
            return
        rel_epoch = epoch - self.warmup + 1
        new_tau = max(self.tau_min, self.tau_init * (self.decay_rate ** rel_epoch))
        self.model.tau.assign(new_tau)
        print(f"tau={new_tau:.4f}")
