import keras as k

class PhysicsLossScheduler(k.callbacks.Callback):
    def __init__(self, hparams, schedule_params):
        super().__init__()
        self.gamma = hparams['gamma']
        self.delta = hparams['delta']
        self.warmup = schedule_params['warmup']
        self.rampup = schedule_params['rampup']
        self.tau_init = schedule_params['tau_init']
        self.tau_min = schedule_params['tau_min']
        self.decay_rate = schedule_params['decay_rate']

    def on_epoch_end(self, epoch, logs=None):
        if epoch < self.warmup:
            return
        rel_epoch = epoch - self.warmup + 1
        ramp_progress = min(1.0, rel_epoch / float(self.rampup))
        self.model.gamma.assign(self.gamma * ramp_progress)
        self.model.delta.assign(self.delta * ramp_progress)
        new_tau = max(self.tau_min, self.tau_init * (self.decay_rate ** rel_epoch))
        self.model.tau.assign(new_tau)
        print(f"tau={new_tau:.4f}, gamma={self.model.gamma.numpy():.4f}, delta={self.model.delta.numpy():.4f}")
