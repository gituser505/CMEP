from keras.callbacks import Callback

from mylib.dataloader import IsingDataLoader
from mylib.model import CVAE
from mylib.callbacks import SoftmaxAnnealing, PhysicalLossScheduler

class GumbelSoftmaxAnnealing(Callback):
    """
    Info
    """
    def __init__(self, init_tau=1.0, min_tau=0.1, decay_rate=0.95):
        super().__init__()
        self.init_tau = init_tau
        self.min_tau = min_tau
        self.decay_rate = decay_rate

    def on_epoch_end(self, epoch, logs=None):
        new_tau = max(self.min_tau, self.init_tau * (self.decay_rate ** (epoch + 1)))
        self.model.tau.assign(new_tau) 
        print(f" - tau annealed to: {new_tau:.4f}")


class PhysicalLossScheduler(Callback):
    """
    Info
    """
    def __init__(self, start_epoch):
        super().__init__()
        self.start_epoch = start_epoch

    def on_epoch_begin(self, epoch, logs=None): 
        if epoch == self.start_epoch:
            self.model.gamma.assign(self.model.target_gamma)
            self.model.delta.assign(self.model.target_delta)
            print(f"\n*** Epoch {epoch + 1}: Physical losses (M & E) are now ACTIVE! ***\n")
