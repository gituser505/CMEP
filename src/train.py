import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import argparse
from pathlib import Path

import json
from datetime import datetime
from matplotlib import pyplot as plt

import numpy as np
import keras as k
from keras.optimizers import Adam, AdamW, Nadam, SGD
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, CSVLogger

from projectlib.dataloader import IsingDataLoader
from projectlib.model_padded import CVAE

ROOT = Path(__file__).parent.parent


class GumbelSoftmaxAnnealing(k.callbacks.Callback):
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


class PhysicalLossScheduler(k.callbacks.Callback):
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
            print(f"\n*** EPOCH {epoch + 1}: Physical losses (M & E) are now ACTIVE! ***\n")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Train CVAE model')
    parser.add_argument('--config', required=True, help='Name of JSON file')
    args = parser.parse_args()
    
    with open(ROOT/"config"/args.config, 'r') as f:
        config = json.load(f)

    hp = config['hyperparams']
    tp = config['train_params']
    
    L = hp['L']
    latent_dim = hp['latent_dim']
    enc_filters = hp['enc_filters']
    mlp_units = hp['mlp_units']
    alpha = hp['alpha']
    
    N = tp['N']
    split_ratio = tp['split_ratio']
    batch_size = tp['batch_size']
    learning_rate = tp['learning_rate']
    min_delta = tp['min_delta']
    epochs = tp['epochs']
    augment = tp['augment']
    data_dir = tp['data_dir']   

    timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    config_str = f"cvae_{latent_dim}_{enc_filters}_{mlp_units}_{alpha}_{batch_size}".replace(" ","")
    config_str = config_str + f"_{timestamp}"
    
    exp_dir = ROOT/"results"/data_dir/config_str
    exp_dir.mkdir(parents=True, exist_ok=True)

    with open(exp_dir/"config.json", 'w') as f:
        json.dump(config, f, indent=2)

    spins_file = ROOT/"data"/data_dir/"lattice_samples.bin"
    betas_file = ROOT/"data"/data_dir/"beta_labels.bin"
    
    loader = IsingDataLoader(spins_file, betas_file, L, N)
    train_data, val_data = loader.get_training_data(split_ratio, batch_size, augment=True)
    
    cvae = CVAE(hp)
    cvae.compile( Adam(learning_rate), jit_compile=True )

    early_stop = EarlyStopping(
        monitor='val_total_loss', 
        mode='min',
        patience=10, 
        min_delta=min_delta, 
        restore_best_weights=True )
    
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss', 
        factor=0.5, 
        patience=5, 
        min_lr=1e-7 )

    checkpoint = ModelCheckpoint(
        str(exp_dir/f"cvae.keras"), 
        save_best_only=True,
        save_weights_only=False, 
        save_freq='epoch', 
        verbose=0)

    #tau_callback = TemperatureAnnealing(init_tau=1.0, min_tau=0.1, decay_rate=0.95)
    physical_loss = PhysicalLossScheduler(start_epoch=5)
    
    history_csv = CSVLogger(exp_dir/f"history.csv", append=True)
    
    callbacks = [early_stop, reduce_lr, history_csv, checkpoint, physical_loss] #, tau_callback]
    
    history = cvae.fit(
        x=train_data,
        validation_data=val_data,
        epochs=epochs,
        callbacks=callbacks,
        verbose=2 )

    cvae.save(str(exp_dir/f"cvae.keras"))
    print(f"\nSaved to: {exp_dir}")

    with open(".latest_run.txt", "w") as f:
        f.write(str(exp_dir))