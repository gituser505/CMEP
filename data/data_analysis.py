from pathlib import Path
import argparse
import numpy as np
import matplotlib.pyplot as plt

path = Path(__file__).parent
parts = path.name.split('_')

L = int(parts[1])
N = int(parts[2])
beta_c = 0.4406868

results = np.loadtxt(path/"results.txt", unpack=True)
results = results[:, np.argsort(results[0])]
beta, tau_M, tau_E, M, M_err, E, E_err, chi, chi_err, C, C_err, bc, bc_err = results

# Scaling transformations
chi[:] = L**2 * beta * chi
chi_err[:] = L**2 * beta * chi_err
C[:] = L**2 * beta**2 * C
C_err[:] = L**2 *beta**2 * C_err

for row in results.T:
    print(", ".join(f"{val:<7.2g}" for val in row))

datasets = [
    (beta, tau_M, tau_E),
    (beta, M, M_err),
    (beta, E, E_err),
    (beta, chi, chi_err),
    (beta, C, C_err),
    (beta, bc, bc_err) ]

configs = [
    ("Correlation", r"$\tau$", (0, 1), r"$\beta$", (0, 0.9)),
    ("Magnetization", r"$|M|$", (0, 1.1), r"$\beta$", (0, 0.9)),
    ("Energy", r"$E$", (-2.1, 0), r"$\beta$", (0, 0.9)),
    ("Susceptibility", r"$\chi$", (0, 20), r"$\beta$", (0, 0.9)),
    ("Specific heat", r"$C$", (0, 3), r"$\beta$", (0, 0.9)),
    ("Binder cumulant", r"$U_L$", (0, 1), r"$\beta$", (0, 0.9)) ]

for (x,y,yerr), (title, ylabel, ylim, xlabel, xlim) in zip(datasets, configs):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_title(title, size=20)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(color='grey', linestyle=':', alpha=0.7)
    ax.set_ylim(ylim)
    ax.set_xlim(xlim)
    if "Correlation" in title:
        ax.plot(x, y, marker='.', linestyle='-')
        ax.plot(x, yerr, marker='.', linestyle='-')
    else:
        ax.errorbar(x, y, yerr, marker='.', linestyle='-')
    plt.savefig(ROOT/f'{title}', dpi=1000)
