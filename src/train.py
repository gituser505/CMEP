import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import argparse
from pathlib import Path

import json
from datetime import datetime

import keras as k
from keras.optimizers import Adam, AdamW
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, CSVLogger

from mylib.dataloader import IsingDataLoader
from mylib.model import CVAE
from mylib.scheduler import PhysicalLossScheduler, GumbelScheduler

ROOT = Path(__file__).parent.parent


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Train CVAE model')
    parser.add_argument('--config', required=True, help='Name of JSON file')
    args = parser.parse_args()
    
    # Load config
    with open(ROOT/"config"/args.config, 'r') as f:
        config = json.load(f)

    # Unpack model, trainig and scheduler parameters
    hp = config['hyperparams']
    tp = config['train_params']
    sp = config['schedule_params']
    
    L = hp['L']
    latent_dim = hp['latent_dim']
    enc_filters = hp['enc_filters']
    mlp_units = hp['mlp_units']
    alpha = hp['alpha']
    gamma = hp['gamma']
    delta = hp['delta']
    use_physics_loss = hp['use_physics_loss']
    use_gumbel = hp['use_gumbel']
    
    N = tp['N']
    split_ratio = tp['split_ratio']
    batch_size = tp['batch_size']
    learning_rate = tp['learning_rate']
    min_delta = tp['min_delta']
    patience = tp['patience']
    epochs = tp['epochs']
    augment = tp['augment']
    data_dir = tp['data_dir']

    warmup = sp['warmup']
    rampup = sp['rampup']

    # Set global keras seed
    k.utils.set_random_seed(hp['seed']) 

    # Build results directory per traning experiment      
    timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    config_str = f"cvae_{latent_dim}_{enc_filters}_{mlp_units}_{batch_size}_{alpha}".replace(" ","")
    
    if use_physics_loss == True:
        config_str += f"_{gamma}_{delta}"
    
    config_str += f"_{timestamp}"
    
    exp_dir = ROOT/"results"/data_dir/config_str
    exp_dir.mkdir(parents=True, exist_ok=True)

    # Save config into results dierectry for later reference
    with open(exp_dir/"config.json", 'w') as f:
        json.dump(config, f, indent=2)

    # Load training/validation
    spins_file = ROOT/"data"/data_dir/"lattice_samples.bin"
    betas_file = ROOT/"data"/data_dir/"beta_labels.bin"
    
    loader = IsingDataLoader(spins_file, betas_file, L, N)
    train_data, val_data = loader.get_training_data(split_ratio, batch_size, augment=True)
    
    # Load and compile model
    cvae = CVAE(hp)
    cvae.compile( Adam(learning_rate), jit_compile=True )

    # Callbacks
    early_stop = EarlyStopping(
        monitor='val_total_loss', 
        mode='min',
        patience=patience,
        start_from_epoch=warmup+rampup,
        min_delta=min_delta, 
        restore_best_weights=True )
    
    reduce_lr = ReduceLROnPlateau(
        monitor='val_total_loss', 
        mode='min',
        factor=0.5, 
        patience=10,
        cooldown=2, 
        min_lr=1e-7 )
    
    checkpoint = ModelCheckpoint(
        str(exp_dir/f"cvae.keras"),
        monitor='val_total_loss', 
        save_best_only=True,
        save_weights_only=False, 
        save_freq='epoch', 
        mode='min',
        verbose=0)

    history_csv = CSVLogger(exp_dir/f"history.csv", append=True)
    callbacks = [early_stop, reduce_lr, checkpoint, history_csv]

    # Add scheduler
    if use_physics_loss == True:
        callbacks.append(PhysicalLossScheduler(hp, sp))
    if use_gumbel == True:
        callbacks.append(GumbelScheduler(sp))
    
    # Run model training
    history = cvae.fit(
        x=train_data,
        validation_data=val_data,
        epochs=epochs,
        callbacks=callbacks,
        verbose=2 )

    # Save model
    cvae.save(str(exp_dir/f"cvae.keras"))
    print(f"\nSaved to: {exp_dir}")

    # Save results path for easy makefile reading
    with open(".latest_run.txt", "w") as f:
        f.write(str(exp_dir))
