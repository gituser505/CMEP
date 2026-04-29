import numba
import numpy as np
from scipy.stats import moment
import matplotlib.pyplot as plt


@numba.njit(parallel=True)
def jackknife_samples(data):
    n = len(data)
    jk_mean = np.empty(n)
    jk_var = np.empty(n)
    jk_bc = np.empty(n)

    sum_m1 = np.sum(data)
    sum_m2 = np.sum(data**2)
    sum_m4 = np.sum(data**4)

    norm = 1.0 / (n - 1)
    
    for i in range(n):
        m1 = (sum_m1 - data[i]) * norm
        m2 = (sum_m2 - data[i]**2) * norm
        m4 = (sum_m4 - data[i]**4) * norm
        
        jk_mean[i] = m1
        jk_var[i] = m2 - m1**2
        jk_bc[i] = 1.0 - m4/(3.0*m2**2)
        
    return jk_mean, jk_var, jk_bc


def jackknife(data):
    n = len(data)
    jk_means, jk_vars, jk_bcs = jackknife_samples(data)
    mean_err  = np.sqrt( (n-1)*np.var(jk_means))
    var_err   = np.sqrt( (n-1)*np.var(jk_vars))
    bc_err    = np.sqrt( (n-1)*np.var(jk_bcs))
    return mean_err, var_err, bc_err


def magnetization(spins):
    M = spins.mean(axis=(1, 2))
    return np.abs(2*M - 1)


def energy(spins, J=1.0):
    s = 2 * spins - 1  # Convert to spin ±1
    right = np.roll(s, shift=-1, axis=2)
    down = np.roll(s, shift=-1, axis=1)
    return -J * np.mean(s * (right + down), axis=(1, 2))


def binder_cumulant(data):
    m2 = moment(data, order=2, center=0)
    if m2 == 0.0: return 0.0
    m4 = moment(data, order=4, center=0)
    return 1.0 - m4/(3.0*m2*m2)


def get_observable_arrays(spins, betas):
    energies = {}
    magnetisations = {}
    for b in sorted(np.unique(betas)):
        mask = np.isclose(betas, b, atol=1e-5)
        spins_at_b = spins[mask] 
        magnetisations[b] = magnetization(spins_at_b)
        energies[b] = energy(spins_at_b)
    return magnetisations, energies


def get_observables(M_arrays, E_arrays, L):
    obs = {
        'M': {'val': [], 'err': []},
        'E': {'val': [], 'err': []},
        'chi': {'val': [], 'err': []},
        'C': {'val': [], 'err': []},
        'bc': {'val': [], 'err': []}
        }
    for beta in M_arrays:
        M_array = M_arrays[beta]
        E_array = E_arrays[beta]
        M_err, chi_err, bc_err = jackknife(M_array)
        E_err, C_err, _ = jackknife(E_array)     
        obs['M']['val'].append(np.mean(M_array))
        obs['M']['err'].append(M_err)
        obs['E']['val'].append(np.mean(E_array))
        obs['E']['err'].append(E_err)
        obs['chi']['val'].append((L**2*beta)*np.var(M_array))
        obs['chi']['err'].append((L**2*beta)*chi_err) 
        obs['C']['val'].append((L**2*beta**2)*np.var(E_array))
        obs['C']['err'].append((L**2*beta**2)*C_err) 
        obs['bc']['val'].append(binder_cumulant(M_array))
        obs['bc']['err'].append(bc_err) 

    for key in obs:
        obs[key]['val'] = np.array(obs[key]['val'])
        obs[key]['err'] = np.array(obs[key]['err'])
       
    return obs


def plot_observables(betas, obs, path):
    configs = [
        ("Magnetization", r"$|M|$", (0, 1.1), r"$\beta$", (0, 0.9)),
        ("Energy", r"$E$", (-2.1, 0), r"$\beta$", (0, 0.9)),
        ("Susceptibility", r"$\chi$", (0, 20), r"$\beta$", (0, 0.9)),
        ("Specific heat", r"$C$", (0, 2), r"$\beta$", (0, 0.9)),
        ("Binder cumulant", r"$U_L$", (0, 1), r"$\beta$", (0, 0.9))
        ]
    for name, (title, ylabel, ylim, xlabel, xlim) in zip(obs, configs):
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.errorbar(betas, obs[name]['val'], obs[name]['err'], label=name, fmt='.-', capsize=3, ecolor='black')
        ax.set_title(title, size=20)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_ylim(ylim)
        ax.set_xlim(xlim)
        ax.legend()
        ax.grid(True, color='grey', linestyle=':', alpha=0.7)
        plt.savefig(path/f'{title}_graph.png', dpi=1000)


def plot_histogram(betas, M_arrays, E_arrays, path=None):
    for beta in betas:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 5))
        ax1.hist(M_arrays[beta], bins='auto', density=True, alpha=0.5, label='M', color='blue')
        ax1.set_title(f'Magnetisation Distribution, $\\beta$ = {beta}')
        ax1.set_xlabel('M')
        ax1.set_ylabel('Pdf')
        ax1.legend()
        ax1.grid(True)
        ax2.hist(E_arrays[beta], bins='auto', density=True, alpha=0.5, label='E', color='blue')
        ax2.set_title(f'Energy Distribution, $\\beta$ = {beta}')
        ax2.set_xlabel('E')
        ax2.set_ylabel('Pdf')
        ax2.legend()
        ax2.grid(True)
        plt.tight_layout()
        plt.savefig(path/f"histogram_{beta}.png", dpi=300)
        

if __name__ == '__main__':
    import argparse
    import json
    from pathlib import Path
    from sklearn.model_selection import train_test_split
    from dataloader import IsingDataLoader

    ROOT = Path(__file__).parent.parent.parent

    parser = argparse.ArgumentParser(description='Train CVAE model')
    parser.add_argument('--config', required=True, help='Path to configuration JSON file')
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = json.load(f)
   
    L = config['hyperparams']['L']   
    N = config['train_params']['N']
    data_dir = config['train_params']['data']

    spins_file = ROOT/"data"/data_dir/"lattice_samples.bin"
    betas_file = ROOT/"data"/data_dir/"beta_labels.bin"   
 
    loader = IsingDataLoader(spins_file, betas_file, L, N)
    
    indices_all = np.arange(N)
    betas_all = loader.get_betas(indices_all)
    betas_unique, beta_category = np.unique(betas_all, return_inverse=True)
    
    print(N)
    print(len(betas_unique))
    print(betas_unique)

    samples_per_beta = 4000

    if samples_per_beta < N/len(betas_all):
        total = samples_per_beta * len(betas_unique)
        indices, _ = train_test_split(indices_all, train_size=total, stratify=beta_category)
    else: 
        indices = indices_all
    
    spins = loader.get_spins(indices)
    betas = loader.get_betas(indices)

    M_arrays, E_arrays = get_observable_arrays(spins, betas)
    
    obs = get_observables(M_arrays, E_arrays, L)
    plot_observables(betas_unique, obs, ROOT/"source")
    
    betas = [0.2, 0.41, 0.42, 0.43, 0.7]
    plot_histogram(betas, M_arrays, E_arrays, ROOT/"source")