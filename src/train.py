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

from mylib.callbacks import GumbelSoftmaxAnnealing, PhysicsLossScheduler
from mylib.dataloader import IsingDataLoader
from mylib.model_padded import CVAE

ROOT = Path(__file__).parent.parent

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Train CVAE model')
    parser.add_argument('--config', required=True, help='Name of JSON file')
    args = parser.parse_args()

    # Load config
    with open(ROOT/"config"/args.config, 'r') as f:
        config = json.load(f)

    # Unpack config dictionary
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

    # Generate unique results string and direcrtory
    timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    config_str = f"cvae_{latent_dim}_{enc_filters}_{mlp_units}_{alpha}_{batch_size}".replace(" ","")
    config_str = config_str + f"_{timestamp}"
    
    exp_dir = ROOT/"results"/data_dir/config_str
    exp_dir.mkdir(parents=True, exist_ok=True)

    # Save config 
    with open(exp_dir/"config.json", 'w') as f:
        json.dump(config, f, indent=2)

    # Load Ising dataset
    spins_file = ROOT/"data"/data_dir/"lattice_samples.bin"
    betas_file = ROOT/"data"/data_dir/"beta_labels.bin"
    
    loader = IsingDataLoader(spins_file, betas_file, L, N)
    train_data, val_data = loader.get_training_data(split_ratio, batch_size, augment=True)

    # Instantiate model and compile
    cvae = CVAE(hp)
    cvae.compile( Adam(learning_rate), jit_compile=True )

    # Callbacks
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

    #tau_annealer = GumbelSoftmaxAnnealing(init_tau=1.0, min_tau=0.1, decay_rate=0.95)
    #physical_loss = PhysicalLossScheduler(start_epoch=25)
    history_csv = CSVLogger(exp_dir/f"history.csv", append=True)
    callbacks = [early_stop, reduce_lr, history_csv] #, physical_loss, tau_annealer]

    # Run training
    history = cvae.fit(
        x=train_data,
        validation_data=val_data,
        epochs=epochs,
        callbacks=callbacks,
        verbose=2 )

    # Save model and weights to results
    cvae.save(str(exp_dir/f"cvae.keras"))
    print(f"\nSaved to: {exp_dir}")

    # Save unique results path to temp for easy lookup
    with open(".latest_run.txt", "w") as f:
        f.write(str(exp_dir))
