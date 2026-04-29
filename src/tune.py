import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '2'

import argparse
from pathlib import Path
import json
from datetime import datetime

import numpy as np
import optuna # NEW IMPORT
from optuna.integration import TFKerasPruningCallback # NEW IMPORT

from keras.optimizers import Adam, AdamW, Nadam
from keras.callbacks import EarlyStopping, ReduceLROnPlateau

from projectlib.dataloader import IsingDataLoader
from projectlib.model_padded import CVAE

ROOT = Path(__file__).parent.parent


def objective(trial, base_config, train_data, val_data):
    """
    Samples hyperparameters, trains the model, and returns the validation loss.
    """
    # Suggest Hyperparameters
    alpha = trial.suggest_float('alpha', 0.1, 10.0, log=True)
    gamma = trial.suggest_float('gamma', 0.01, 100.0, log=True)
    delta = trial.suggest_float('delta', 0.01, 100.0, log=True)
    latent_exp = trial.suggest_int('latent_exp', 1, 8)
    mlp_units_exp = trials.suggest_int('mlp_units_exp', 5, 9)
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True)
    
    # Update config with suggested values
    hp = base_config['hyperparams'].copy()
    tp = base_config['train_params'].copy()
   
    hp['alpha'] = alpha
    hp['gamma'] = gamma
    hp['delta'] = delta
    hp['latent_dim'] = 2**latent_exp
    tp['learning_rate'] = learning_rate

    # Static parameters
    min_delta = tp['min_delta']
    epochs = tp['epochs']
    
    # Build and compile model
    cvae = CVAE(hp)
    cvae.compile(Adam(learning_rate), jit_compile=True)

    # Callbacks
    early_stop = EarlyStopping(
        monitor='val_unweighted_loss', 
        mode='min',
        patience=10, 
        min_delta=min_delta, 
        restore_best_weights=True)
    
    reduce_lr = ReduceLROnPlateau(
        monitor='val_unweighted_loss', 
        mode='min',
        factor=0.5,
        patience=5,
        min_lr=1e-7)
    
    pruning = TFKerasPruningCallback(trial, monitor='val_unweighted_loss')
    
    callbacks = [early_stop, reduce_lr, pruning]
    
    # Train model
    history = cvae.fit(
        x=train_data, 
        validation_data=val_data, 
        epochs=epochs, 
        callbacks=callbacks, 
        verbose=2)

    # Return the best validation loss achieved in this trial
    return min(history.history['val_unweighted_loss'])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Tune CVAE model')
    parser.add_argument('--config', required=True, help='Name of base JSON file')
    args = parser.parse_args()

    # Load config
    with open(ROOT / "config" / args.config, 'r') as f:
        config = json.load(f)

    # Unpack config
    L = config['hyperparams']['L']
    N = config['train_params']['N']
    split_ratio = config['train_params']['split_ratio']
    batch_size = config['train_params']['batch_size']
    trials = config['train_params']['trials']
    data_dir = config['train_params']['data_dir']  
    
    # Define dynamic data paths
    spins_file = ROOT/"data"/data_dir/"lattice_samples.bin"
    betas_file = ROOT/"data"/data_dir/"beta_labels.bin"
    
    # Generate training datasets
    loader = IsingDataLoader(spins_file, betas_file, L, N)
    train_data, val_data = loader.get_training_data(split_ratio, batch_size, buffer_size=N, augment=False)

    print("\nLoaded training and validation data.")
    print(f"\nStarting Hyperparameter Tuning for {trials} trials...")
    
    # Create an Optuna study and "minimize" val_loss
    study = optuna.create_study(
        direction="minimize", 
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=5) )
    
    # Run the optimization
    study.optimize(lambda trial: objective(trial, config, train_data, val_data), n_trials=trials )

    # Save study history and best_config
    results_dir = ROOT/"results"/data_dir/"tuning"
    results_dir.mkdir(parents=True, exist_ok=True)

    best = study.best_trial
    df = study.trials_dataframe()
    df.to_csv(results_dir/"all_trials_summary.csv", index=False)

    with open(results_dir/"base_config.json", 'w') as f:
        json.dump(config, f, indent=2)
        
    best_config = config.copy()
    best_config['hyperparams']['alpha'] = best.params['alpha']
    best_config['hyperparams']['gamma'] = best.params['gamma']
    best_config['hyperparams']['delta'] = best.params['delta']
    best_config['hyperparams']['latent_dim'] = best.params['latent_dim']
    best_config['train_params']['learning_rate'] = best.params['learning_rate']    
    
    best_config_path = ROOT/"config"/f"best_config_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(best_config_path,  'w') as f:
        json.dump(best_config, f, indent=2)
    
    print(f"\nBest parameters saved to: {best_config_path}")

    # Print the final results
    print("\n--- Tuning Completed ---")
    print("Best Trial:")
    print(f"  Value (Val Loss): {best.value}")
    print("  Best Parameters:")
    for key, value in best.params.items():
        print(f"{key}: {value}")