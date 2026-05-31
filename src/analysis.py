import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

from pathlib import Path
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import keras as k
from keras.models import load_model
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

from mylib.dataloader import IsingDataLoader
from mylib.model_padded import CVAE
from mylib.observables import *

ROOT = Path(__file__).parent.parent


def norm_array(x):
    min, max = x.min(), x.max()
    return (x-min)/(max-min)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Analyse trained CVAE model')
    parser.add_argument('--dir', type=str, required=True, help='model directory (e.g. results/exp_...)')
    args = parser.parse_args()

    results_dir = ROOT/"results"/args.dir

    # Load config
    with open(results_dir/"config.json", 'r') as f:
        config = json.load(f)

    # Unpack parameters 
    L = config['hyperparams']['L']
    N = config['train_params']['N']
    data_dir = config['train_params']['data_dir']

    # Get data
    spins_file = ROOT/"data"/data_dir/"lattice_samples.bin"
    betas_file = ROOT/"data"/data_dir/"beta_labels.bin"   
 
    loader = IsingDataLoader(spins_file, betas_file, L, N)

    all_indices = np.arange(N)
    betas = loader.get_betas(all_indices)
    betas_norm = norm_array(betas)
    spins = loader.get_spins(all_indices)

    betas_unique, beta_category = np.unique(betas, return_inverse=True)

    # Get reference ising model observables and thermodynamic averages
    M_ising, E_ising = get_observable_arrays(spins, betas)
    obs_ising = get_observables(M_ising, E_ising, L)

    # Load model
    cvae = load_model(results_dir/"cvae.keras")

    # Get CVAE generated output and observables
    batch_size = 5000
    spins_cvae_list = []

    for i in range(0, len(betas_norm), batch_size):
        betas_batch = betas_norm[i : i + batch_size]
        spins_batch = cvae.generate(betas_batch, stochastic=True)
        spins_cvae_list.append(spins_batch.numpy().squeeze(-1))
    spins_cvae = np.concatenate(spins_cvae_list, axis=0)

    M_cvae, E_cvae = get_observable_arrays(spins_cvae, betas)
    obs_cvae = get_observables(M_cvae, E_cvae, L)

    """
    spins_mode_list = []

    for i in range(0, len(betas_norm), batch_size):
        betas_batch = betas_norm[i : i + batch_size]
        spins_batch = cvae.generate(betas_batch, stochastic=False)
        spins_mode_list.append(spins_batch.numpy().squeeze(-1))
    spins_mode = np.concatenate(spins_mode_list, axis=0)

    M_mode, E_mode = get_observable_arrays(spins_mode, betas)
    obs_mode = get_observables(M_mode, E_mode, L)
    """
    
    # Get continuous output and observables for comparison
    spins_float_list = []
    for i in range(0, len(betas_norm), batch_size):
        betas_batch = betas_norm[i : i + batch_size]
        num_samples = len(betas_batch)
        betas_batch = k.ops.cast(k.ops.reshape(betas_batch, [-1, 1]), "float32")
        z = k.random.normal([num_samples, cvae.latent_dim])       
        spins_gen = cvae.decoder([z, betas_batch])
        spins_float_list.append(k.ops.sigmoid(spins_gen).numpy().squeeze(-1))
    spins_float = np.concatenate(spins_float_list, axis=0)

    M_float, E_float = get_observable_arrays(spins_float, betas)
    obs_float = get_observables(M_float, E_float, L)

   
    # --- PLOT1: training losses ---
    history = pd.read_csv(results_dir/"history.csv")
    losses = [col for col in history.columns if not col.startswith('val_') and col not in ['epoch', 'learning_rate']]
    for loss in losses:
        fig, ax = plt.subplots(figsize=(10, 6), layout="constrained")
        ax.plot(history['epoch'], history[loss], label='Training', color='blue', linestyle='-')
        ax.plot(history['epoch'], history['val_'+loss], label='Validation', color='orange', linestyle='--')       
        ax.set_title(loss.replace('_', ' ').title(), fontsize=16)
        ax.set_xlabel("Epoch", fontsize=14)
        ax.set_ylabel("Loss", fontsize=14)
        ax.set_xlim()
        ax.set_ylim()
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend(fontsize=12)
        plt.savefig(results_dir/f"{loss}.png", dpi=300, bbox_inches='tight')
        plt.close(fig)


    # --- PLOT 2: PCA ---
    samples_per_beta = 100
    total_pca = samples_per_beta * len(betas_unique)
    indices_pca, _ = train_test_split(all_indices, train_size=total_pca, stratify=beta_category)
    spins_pca = loader.get_spins(indices_pca)
    betas_pca = loader.get_betas(indices_pca)

    M_pca = 2*np.mean(spins_pca, axis=(1,2)) - 1
    E_pca = energy(spins_pca)
    
    betas_norm_pca_input = norm_array(betas_pca).reshape(-1,1)
    spins_pca_input = spins_pca.reshape((-1,L,L,1)).astype(np.float32)

    z_mean, _, _ = cvae.encoder.predict([spins_pca_input, betas_norm_pca_input])

    pca = PCA()
    z_pca = pca.fit_transform(z_mean)
    var_ratios = pca.explained_variance_ratio_
    
    # PCA colored by beta
    fig, ax = plt.subplots(figsize=(8, 6), layout="constrained")
    scatter = ax.scatter(z_pca[:,0], z_pca[:,1], c=betas_pca, cmap='viridis', s=5, alpha=0.7)
    fig.colorbar(scatter, ax=ax, label='Beta value')
    ax.set_title('PCA of Latent Means')
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.grid(True, alpha=0.3)
    plt.savefig(results_dir/'latent_pca.png', dpi=500)
    plt.close(fig)
    
    # Cumulative variance  
    fig, ax = plt.subplots(figsize=(8, 6), layout="constrained")
    ax.plot(var_ratios.cumsum() * 100, '-')
    ax.set_xlabel('Principal Component')
    ax.set_ylabel('Cumulative Variance (%)')
    ax.set_title('PCA Cumulative Variance')
    ax.grid(True, alpha=0.3)
    plt.savefig(results_dir/'latent_pca_cumulative_varinace.png', dpi=500)
    plt.close(fig)

    # PCA-observables correlation ---
    fig, ax = plt.subplots(2, 2, figsize=(12, 5), layout='constrained')
    
    sc1 = ax[0,0].scatter(z_pca[:,0], M_pca, c=betas_pca, cmap='coolwarm', s=10, alpha=0.6)
    ax[0,0].set_xlabel("PCA Component 1")
    ax[0,0].set_ylabel("Magnetization $M$")
    ax[0,0].set_title("PCA 1 vs Magnetization")
    fig.colorbar(sc1, ax=ax[0,0], label=r'$\beta$')
    
    sc2 = ax[0,1].scatter(z_pca[:,1], E_pca, c=betas_pca, cmap='coolwarm', s=10, alpha=0.6)
    ax[0,1].set_xlabel("PCA Component 2")
    ax[0,1].set_ylabel("Energy $E$")
    ax[0,1].set_title("PCA 2 vs Energy")
    fig.colorbar(sc2, ax=ax[0,1], label=r'$\beta$')
    
    sc3 = ax[1, 0].scatter(z_pca[:,0], E_pca, c=betas_pca, cmap='coolwarm', s=10, alpha=0.6)
    ax[1, 0].set_xlabel("PCA Component 1")
    ax[1, 0].set_ylabel(r"$E$")
    ax[1, 0].set_title(r"PCA 1 vs $E$")
    fig.colorbar(sc3, ax=ax[1,0], label=r"$\beta$")
    
    sc4 = ax[1, 1].scatter(z_pca[:,1], M_pca, c=betas_pca, cmap='coolwarm', s=10, alpha=0.6)
    ax[1, 1].set_xlabel("PCA Component 2")
    ax[1, 1].set_ylabel(r"$M$")
    ax[1, 1].set_title(r"PCA 2 vs $M$")
    fig.colorbar(sc4, ax=ax[1,1], label=r"$\beta$")
    
    plt.savefig(results_dir / "pca_vs_physics.png")
    plt.close(fig)
    

    # --- PLOT 3: Observables ---
    configs = [
        ("Magnetization", r"$|M|$", (0, 1.1), r"$\beta$", (0, 0.9)),
        ("Energy", r"$E$", (-2.1, 0), r"$\beta$", (0, 0.9)),
        ("Susceptibility", r"$\chi$", (0, 20), r"$\beta$", (0, 0.9)),
        ("Specific heat", r"$C$", (0, 2), r"$\beta$", (0, 0.9)),
        ("Binder cumulant", r"$U_L$", (0, 1), r"$\beta$", (0, 0.9))
        ]
    for name, (title, ylabel, ylim, xlabel, xlim) in zip(obs_ising, configs):
        fig, ax = plt.subplots(figsize=(8, 6), layout="constrained")
        ax.errorbar(betas_unique, obs_ising[name]['val'], obs_ising[name]['err'], label=name, fmt='.-', capsize=3, ecolor='black')
        ax.errorbar(betas_unique, obs_cvae[name]['val'], obs_cvae[name]['err'], label=name, fmt='.-', capsize=3, ecolor='black')
        #ax.errorbar(betas_unique, obs_float[name]['val'], obs_float[name]['err'], label=name, fmt='.-', capsize=3, ecolor='black')
        #ax.errorbar(betas_unique, obs_mode[name]['val'], obs_mode[name]['err'], label=name, fmt='.-', capsize=3, ecolor='black')
        ax.set_title(title, size=20)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_ylim(ylim)
        ax.set_xlim(xlim)
        ax.legend()
        ax.grid(True, color='grey', linestyle=':', alpha=0.7)
        plt.savefig(results_dir/f'{title}.png', dpi=500)
        plt.close(fig)


    # --- Histograms ---    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10), layout="constrained")

    for i, beta in enumerate([0.2, 0.44, 0.7]):
        M_combined = np.concatenate([M_ising[beta], M_cvae[beta]])
        E_combined = np.concatenate([E_ising[beta], E_cvae[beta]])
        M_bins = np.histogram_bin_edges(M_combined, bins='auto')
        E_bins = np.histogram_bin_edges(E_combined, bins='auto')

        ax_m = axes[0, i]
        ax_m.hist(M_ising[beta], bins=M_bins, density=True, alpha=0.5, label='Ising', color='blue')
        ax_m.hist(M_cvae[beta], bins=M_bins, density=True, alpha=0.5, label='CVAE', color='orange')
        #ax_m.hist(M_mode[beta], bins=M_bins, density=True, alpha=0.5, label='Mode', color='green')
        #ax_m.hist(M_float[beta], bins=M_bins, density=True, alpha=0.5, label='Float', color='red')
        ax_m.axvline(np.mean(M_ising[beta]), color='blue', linestyle=':', label=f'Ising Mean {np.mean(M_ising[beta]):.3}')
        ax_m.axvline(np.mean(M_cvae[beta]), color='orange', linestyle=':', label=f'CVAE Mean {np.mean(M_cvae[beta]):.3}')
        ax_m.set_title(f'Magnetization ($\\beta={beta}$)')
        ax_m.set_xlabel('M')
        ax_m.set_ylabel('Density')
        ax_m.legend()
        ax_m.grid(True)

        ax_e = axes[1, i]
        ax_e.hist(E_ising[beta], bins=E_bins, density=True, alpha=0.5, label='Ising', color='blue')
        ax_e.hist(E_cvae[beta], bins=E_bins, density=True, alpha=0.5, label='CVAE', color='orange')
        #ax_m.hist(E_mode[beta], bins=E_bins, density=True, alpha=0.5, label='Mode', color='green')
        #ax_m.hist(E_float[beta], bins=M_bins, density=True, alpha=0.5, label='Float', color='red')
        ax_e.axvline(np.mean(E_ising[beta]), color='blue', linestyle=':', label=f'Ising Mean {np.mean(E_ising[beta]):.3}')
        ax_e.axvline(np.mean(E_cvae[beta]), color='orange', linestyle=':', label=f'CVAE Mean {np.mean(E_cvae[beta]):.3}')
        ax_e.set_title(f'Energy ($\\beta={beta}$)')
        ax_e.set_xlabel('E')
        ax_e.set_ylabel('Density')
        ax_e.legend()
        ax_e.grid(True)

    plt.savefig(results_dir/"histograms.png", dpi=300)
    plt.close(fig)


    # --- PLOT 5: Reconstructions ---    
    betas_to_plot = [0.2, 0.44, 0.7]
    titles = ["Input", "Recon", "Recon (threshold)", "Recon (stochastic)", "Generated (threshold)", "Generated (stochastic)"]

    fig, axes = plt.subplots(len(betas_to_plot), 6, figsize=(12, 6), layout="constrained")

    for row, b in enumerate(betas_to_plot): 
        mask = np.isclose(betas.flatten(), b, atol=1e-5)
        spin = spins[mask][:1].reshape((1, L, L, 1)) 
        beta = betas[mask][:1].reshape((1, 1))
        beta_norm = betas_norm[mask][:1].reshape((1, 1))
        
        spin_recon = cvae.predict([spin, beta_norm], verbose=0)[0]
        spin_recon = k.ops.sigmoid(spin_recon).numpy().squeeze()
        spin_recon_det = np.where(spin_recon >= 0.5, 1.0, 0.0).astype(np.int64)
        spin_recon_stoch = np.random.binomial(n=1, p=spin_recon).astype(np.int64) 
        spin_mode = cvae.generate(beta_norm, stochastic=False).numpy().squeeze()
        spin_random = cvae.generate(beta_norm, stochastic=True).numpy().squeeze()
        spin = spin.squeeze()

        datasets = [spin, spin_recon, spin_recon_det, spin_recon_stoch, spin_mode, spin_random]
        
        # 5. Plot across the 6 columns for the current row
        for col, (data, title) in enumerate(zip(datasets, titles)):
            ax = axes[row, col]
            if row == 0: ax.set_title(title, fontsize=12)
            for spine in ax.spines.values():
                spine.set_edgecolor('black')
                spine.set_linewidth(2)
            ax.imshow(data, cmap='gray', vmin=0, vmax=1)
            ax.set_xticks([])
            ax.set_yticks([])
            if col == 0: ax.set_ylabel(f"$\\beta$ = {b}", fontsize=14, fontweight='bold')
                
    fig.suptitle("Lattice Comparison", fontsize=16, fontweight='bold')
    plt.savefig(results_dir/"reconstructions_combined.png", dpi=500, bbox_inches='tight')
    plt.close(fig)


    # --- PLOT 6: 10 Generative Samples per Temperature ---    
    betas_to_plot = [0.2, 0.44, 0.7]
    num_lattices = 10

    fig, axes = plt.subplots(len(betas_to_plot), num_lattices, figsize=(18, 5), layout="constrained")

    for row, b in enumerate(betas_to_plot): 
        mask = np.isclose(betas.flatten(), b, atol=1e-5)
        single_beta_norm = betas_norm[mask][0] 
        beta_batch = np.repeat(single_beta_norm, num_lattices).reshape((num_lattices, 1))
        spins_gen = cvae.generate(beta_batch, stochastic=True).numpy().squeeze(-1)

        # 5. Plot the 10 generated lattices across the columns
        for col in range(num_lattices):
            ax = axes[row, col]
            
            # Draw bold black borders around each image
            for spine in ax.spines.values():
                spine.set_edgecolor('black')
                spine.set_linewidth(2)
                
            # Plot the specific generated lattice for this column
            ax.imshow(spins_gen[col], cmap='gray', vmin=0, vmax=1)
            ax.set_xticks([])
            ax.set_yticks([])
            
            # Set column and row titles (row/column 0 only)
            if row == 0: 
                ax.set_title(f"Sample {col+1}", fontsize=12)
            if col == 0: 
                ax.set_ylabel(f"$\\beta$ = {b}", fontsize=16, fontweight='bold')
                
    fig.suptitle("Generative Samples", fontsize=18, fontweight='bold')
    plt.savefig(results_dir/"recon_samples.png", dpi=300, bbox_inches='tight')
    plt.close(fig)


    # --- Latent traversal ---
    num_steps = 100
    betas_traversal = np.linspace(0.1, 0.8, num_steps).reshape((num_steps, 1))
    betas_traversal_norm = norm_array(betas_traversal)

    # Generate and duplicate  random latent vector
    z_single = np.random.normal(size=(1, cvae.latent_dim)).astype(np.float32)
    z_fixed = np.repeat(z_single, repeats=num_steps, axis=0)

    logits = cvae.decoder.predict([z_fixed, betas_traversal_norm], verbose=0)
    spins_probs = 1.0 / (1.0 + np.exp(-logits))

    r = np.random.uniform(spins_probs.shape)
    spins_traversal = (spins_probs >= 0.5).astype(np.int8)
    spins_traversal_stochastic = (spins_probs > r).astype(np.int8)

    M = magnetization(spins_traversal)
    M_s = magnetization(spins_traversal_stochastic)

    # --- Traversal Plot ---
    plt.figure(figsize=(8, 6))
    plt.plot(betas_traversal, M_s, marker='.', linestyle='-', color='orange', label='stochastic')
    plt.plot(betas_traversal, M, marker='.', linestyle='-', color='red', label='Fixed')
    plt.plot(betas_unique, obs_ising['M']['val'], linestyle='-')
    plt.title("Latent Traversal: Magnetization vs Beta (Fixed z)", fontsize=16)
    plt.xlabel("Beta", fontsize=14)
    plt.ylabel("|M|", fontsize=14)
    plt.grid(True, linestyle='--')
    plt.legend()
    plt.savefig(results_dir/f"latent_traversal.png", dpi=500, bbox_inches='tight')
    plt.close(fig)

