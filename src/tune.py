import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '2'

import argparse
from pathlib import Path
import json
from datetime import datetime

import numpy as np
import optuna
from optuna.integration import TFKerasPruningCallback

from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ReduceLROnPlateau

from mylib.dataloader import IsingDataLoader
from mylib.model_padded import CVAE
from mylib.schedulers import PhysicsScheduler, GumbelScheduler
from mylib.observables import magnetization, energy
from mylib.observables import get_observable_arrays

ROOT = Path(__file__).parent.parent


def get_baseline(val_data):
    """Extracts baseline physical observables from the validation dataset.

     Args:
        val_data (tf.data.Dataset): The validation dataset containing batches of 
            (spins, betas_norm) tensors.

    Returns:
        tuple: A tuple containing:
            - observables (dict): A dictionary mapping unique beta values to their 
              corresponding true 'M' (magnetization) and 'E' (energy) values.
            - samples_per_beta (int): The number of lattice samples per unique beta.
    """
    all_spins = []
    all_betas_norm = []
    
    # Extract all data from the validation dataset
    for spins, betas_norm in val_data:
        all_spins.append(spins.numpy())
        all_betas_norm.append(betas_norm.numpy())
        
    val_spins = np.concatenate(all_spins, axis=0).squeeze(-1)
    val_betas_norm = np.concatenate(all_betas_norm, axis=0).squeeze(-1)

    betas_norm_unique = np.unique(np.round(val_betas_norm, decimals=5))
    samples_per_beta = len(val_betas_norm)//len(betas_norm_unique)
    
    # Compare ising baseline to cvae output
    observables = {}
    for b in sorted(betas_norm_unique):
        mask = np.isclose(val_betas_norm, b, atol=1e-5)
        spins_at_b = val_spins[mask]
        true_M = np.mean(magnetization(spins_at_b))
        true_E = np.mean(energy(spins_at_b))
        observables[b] = {'M': true_M, 'E': true_E}

    return observables, samples_per_beta


def suggestions(trial, search_space_dict):
    """Dynamically samples hyperparameters using Optuna based on a configuration.

    Args:
        trial (optuna.trial.Trial): The current Optuna trial object.
        search_space_dict (dict): A dictionary defining the hyperparameter search 
            space. Keys starting with an underscore are ignored.

    Returns:
        dict: A dictionary of sampled hyperparameter values for the current trial.
    """
    suggested = {}
    for key, settings in search_space_dict.items():
        if key.startswith("_"):
            continue
        
        if settings["type"] == "float":
            suggested[key] = trial.suggest_float(
                key, settings["low"], settings["high"], log=settings.get("log", False))
        elif settings["type"] == "int":
            suggested[key] = trial.suggest_int(
                key, settings["low"], settings["high"], log=settings.get("log", False))
            
    return suggested


def objective(trial, base_config, train_data, val_data, ising_obs, samples_per_beta):
    """The Optuna objective function for tuning the CVAE model.

    This function manages a complete training lifecycle for a single trial:
    it sets up the hyperparameters, compiles the model, trains it, and evaluates 
    its generative capability according to a physics loss function:
    $\text{error} = \sum_{\beta} \left( \frac{|M_{true} - M_{gen}|}{|M_{true}|} + \frac{|E_{true} - E_{gen}|}{|E_{true}|} \right)$

    Args:
        trial (optuna.trial.Trial): The current Optuna trial object.
        base_config (dict): The base configuration dictionary containing standard 
            hyperparameters, training parameters, and scheduler settings.
        train_data (tf.data.Dataset): The dataset used for training.
        val_data (tf.data.Dataset): The dataset used for validation.
        ising_obs (dict): Baseline physical observables computed from `get_baseline`.
        samples_per_beta (int): The number of samples to generate per beta for evaluation.

    Returns:
        float: The total accumulated physical error (relative deviation from the baseline).
    """
    hp = base_config['hyperparams'].copy()
    tp = base_config['train_params'].copy()
    sp = base_config['schedule_params'].copy()
    
    # Dynamically load suggested values if a search space is defined
    search_space = base_config['search_space']
    suggested_params = suggestions(trial, search_space)
    
    # Sort suggestions into hp or tp
    for key, value in suggested_params.items():
        if key in hp:
            hp[key] = value
        elif key in tp:
            tp[key] = value

    # Unpack parameters    
    use_physics_loss = hp['use_physics_loss']
    use_gumbel = hp['use_gumbel']
    start_from_epoch = tp['start_from_epoch']
    learning_rate = tp['learning_rate']
    min_delta = tp['min_delta']
    patience = tp['patience']
    epochs = tp['epochs']

    # Print trial paramter values
    for key, value in trial.params.items():
        print(f"  {key}: {value}")
    
    # Build and compile model
    cvae = CVAE(hp)
    cvae.compile(Adam(learning_rate), jit_compile=True)

    # Callbacks
    early_stop = EarlyStopping(
        monitor='val_physics_loss', 
        mode='min',
        patience=patience, 
        min_delta=min_delta,
        start_from_epoch=start_from_epoch, 
        restore_best_weights=True )
    
    reduce_lr = ReduceLROnPlateau(
        monitor='val_physics_loss', 
        mode='min',
        factor=0.5, 
        patience=10, 
        cooldown=2,
        min_lr=1e-7 )
    
    pruning = TFKerasPruningCallback(trial, monitor='val_physics_loss')
    callbacks = [early_stop, reduce_lr, pruning]

    # Add schedulers
    if use_physics_loss == True:
        callbacks.append(PhysicsScheduler(hp, sp))
    if use_gumbel == True:
        callbacks.append(GumbelScheduler(sp))

    # Train model
    history = cvae.fit(
        x=train_data, 
        validation_data=val_data, 
        epochs=epochs, 
        callbacks=callbacks, 
        verbose=2)

    # Compute tuning loss
    betas_norm_unique = list(ising_obs.keys())
    betas_norm_array = np.repeat(betas_norm_unique, samples_per_beta)
    
    gen_spins = cvae.generate(betas_norm_array, stochastic=True).numpy().squeeze(-1)
    
    physics_error = 0.0
    for b in sorted(betas_norm_unique):
        mask = np.isclose(betas_norm_array, b, atol=1e-5)
        b_spins = gen_spins[mask]
        gen_M = np.mean(magnetization(b_spins))
        gen_E = np.mean(energy(b_spins))
        true_M = ising_obs[b]['M']
        true_E = ising_obs[b]['E']
        err_M = np.abs(true_M - gen_M)/np.abs(true_M)
        err_E = np.abs(true_E - gen_E)/np.abs(true_E)
        physics_error += err_M + err_E

    return physics_error


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
    train_data, val_data = loader.get_training_data(split_ratio, batch_size, augment=False)
    print("\nLoaded training and validation data.")

    # Get baseline observables for tuning
    ising_obs, samples_per_beta = get_baseline(val_data)

    # Implement Optuna hyperparameter search
    print(f"\nStarting Hyperparameter Tuning for {trials} trials...")
    
    study = optuna.create_study(
        direction="minimize", 
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=5) )
    
    study.optimize(lambda trial: objective(trial, config, train_data, val_data, ising_obs, samples_per_beta), n_trials=trials )

    # Save study history
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_dir = ROOT/"results"/data_dir/f"tuning_{timestamp}"
    results_dir.mkdir(parents=True, exist_ok=True)

    best = study.best_trial
    df = study.trials_dataframe()
    df.to_csv(results_dir/"trials_summary.csv", index=False)

    with open(results_dir/"base_config.json", 'w') as f:
        json.dump(config, f, indent=2)
        
    # Map the best parameters back into the config sections
    best_config = config.copy()
    search_space = best_config.get('search_space', {})
    
    for key, value in best.params.items():
        if key in best_config['hyperparams']:
            best_config['hyperparams'][key] = value
        if key in best_config['train_params']:
            best_config['train_params'][key] = value
              
    # Save optimal config
    best_config_path = ROOT/"config"/f"best_config_{timestamp}.json"
    with open(best_config_path, 'w') as f:
        json.dump(best_config, f, indent=2)

    # Save results path for easy makefile reading
    with open(".latest_tune.txt", "w") as f:
        f.write(str(results_dir))
    
    print(f"\nBest parameters saved to: {best_config_path}")
    print("\n--- Tuning Completed ---")
    print(f"Best Value (Val Loss): {best.value}")
    print("Best Parameters:")
    for key, value in best.params.items():
        print(f"  {key}: {value}")
