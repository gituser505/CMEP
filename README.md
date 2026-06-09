# CVAE for Ising Model Lattice Generation

A conditional variational autoencoder (CVAE) designed to generate synthetic 2D Ising model lattices.
The model is trained to capture the phase transition dynamics and generate physically realistic configurations from which thermodynamic observables can be calculated (like magnetisation, energy, etc.).

## Overview
This project uses **Keras** and **TensorFlow** to implement a generative model that learns the Boltzmann distribution of a 2d Ising lattice, composed of 32x32 binary spins (0/1). Unlike a standard VAE, this CVAE is conditioned on the temperature (or inverse temperature, $\beta$).

## Tech Stack
- **Python 3.x**
- **Keras 3.x / Tensorflow 2.20.x** (Backend)
- **NumPy & Matplotlib** (Data processing and visualization)
- Ising model souce code in C (Used for initial data generation)

## Features
- **Conditional Generation:** Generate lattices at specific temperatures (beta).
- **Physical Validation:** Includes scripts to calculate Energy, Magnetization, and Specific Heat of generated samples to compare
    between the monte-carlo and the CVAE generated data.
- **Lattice Visualization:** Tools to plot the spin configurations.

## Installation
1. Clone the repo:
   `git clone https://github.com/user505/CMEP`
2. Install dependencies:
   `pip install -r requirements.txt`

## Usage
1. **Generate Data:** Run `make` in the ising directory to create the training lattices via the Wolff algorithm. All data is saved in the data directory.
2. **Train:** Run `python3 train.py --config config.json` in src directory to train the model with the custom configuration file saved in the config directory.
3. **Tune:** Run `python3 tune.py --config config.json` in src directory for hyperparameter tuning with custom configuration file saved in the config directory.
4. **Inference:** Run `python3 analysis.py --dir "results path"` in src to produce graphs of the model behaviour. Path of the results saved in the results directory

5. To run the whole pipeline together (except data generation), run `make` in the root CEMP directory.
