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
from mylib.model_padded import CVAE
from mylib.schedulers import PhysicsScheduler, GumbelScheduler

ROOT = Path(__file__).parent.parent


def load_config(config_path):
    """
    Docstring for load_config
    
    :param config_path: Description
    """
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config


def save_config(config, path):
    """
    Docstring for save_config
    
    :param config: Description
    :param path: Description
    """
    with open(path/"config.json", 'w') as f:
        json.dump(config, f, indent=2)


def build_results_dir(config):
    """Build results directory per traning experiment
    """  
    hp = config['hyperparams']
    tp = config['train_params']
    
    latent_dim = hp['latent_dim']
    enc_filters = hp['enc_filters']
    mlp_units = hp['mlp_units']
    alpha = hp['alpha']
    gamma = hp['gamma']
    delta = hp['delta']
    use_physics_loss = hp['use_physics_loss']
    batch_size = tp['batch_size']
    data_dir = tp['data_dir']

    timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    config_str = f"cvae_{latent_dim}_{enc_filters}_{mlp_units}_{batch_size}_{alpha}".replace(" ","")
    
    if use_physics_loss == True:
        config_str += f"_{gamma}_{delta}"
    
    config_str += f"_{timestamp}"
    
    results_dir = ROOT/"results"/data_dir/config_str
    results_dir.mkdir(parents=True, exist_ok=True)

    return results_dir


def build_callbacks(config, results_dir):
    """
    Docstring for build_callbacks
    
    :param config: Description
    :param results_dir: Description
    """
    # Unpack model, trainig and scheduler parameters
    hp = config['hyperparams']
    tp = config['train_params']
    sp = config['schedule_params']
    
    use_physics_loss = hp['use_physics_loss']
    use_gumbel = hp['use_gumbel']
    min_delta = tp['min_delta']
    patience = tp['patience']
    warmup = sp['warmup']
    rampup = sp['rampup']

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
        str(results_dir/f"cvae.keras"),
        monitor='val_total_loss', 
        save_best_only=True,
        save_weights_only=False, 
        save_freq='epoch', 
        mode='min',
        verbose=0)

    history_csv = CSVLogger(results_dir/f"history.csv", append=True)
    callbacks = [early_stop, reduce_lr, checkpoint, history_csv]

    # Add scheduler
    if use_physics_loss == True:
        callbacks.append(PhysicsScheduler(hp, sp))
        if use_gumbel == True:
            callbacks.append(GumbelScheduler(sp))
    
    return callbacks


def main():
    """
    Docstring for main
    """
    parser = argparse.ArgumentParser(description='Train CVAE model')
    parser.add_argument('--config', required=True, help='Name of JSON file')
    args = parser.parse_args()

    config = load_config(ROOT/"config"/args.config)

    results_dir = build_results_dir(config)

    save_config(config, results_dir)

    # Unpack model, trainig and scheduler parameters
    hp = config['hyperparams']
    tp = config['train_params']
    sp = config['schedule_params']
    
    L = hp['L']   
    N = tp['N']
    split_ratio = tp['split_ratio']
    batch_size = tp['batch_size']
    learning_rate = tp['learning_rate']
    epochs = tp['epochs']
    augment = tp['augment']
    data_dir = tp['data_dir']

    # Set global keras seed
    k.utils.set_random_seed(hp['seed']) 

    # Load training/validation
    spins_file = ROOT/"data"/data_dir/"lattice_samples.bin"
    betas_file = ROOT/"data"/data_dir/"beta_labels.bin"
    
    loader = IsingDataLoader(spins_file, betas_file, L, N)
    train_data, val_data = loader.get_training_data(split_ratio, batch_size, augment=augment)
    
    # Load and compile model
    cvae = CVAE(hp)
    cvae.compile( Adam(learning_rate), jit_compile=True )

    # Apply callbacks
    callbacks = build_callbacks(config, results_dir)

    # Run model training
    history = cvae.fit(
        x=train_data,
        validation_data=val_data,
        epochs=epochs,
        callbacks=callbacks,
        verbose=2 )
    
    # Save model
    cvae.save(str(results_dir/f"cvae.keras"))
    print(f"\nSaved to: {results_dir}")

    # Save results path for easy makefile reading
    with open(".latest_run.txt", "w") as f:
        f.write(str(results_dir))


if __name__ == "__main__":
    main()
