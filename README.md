# CVAE for Ising Model Lattice Generation

A conditional variational autoencoder (CVAE) designed to generate synthetic 2D Ising model lattices.
The model is trained to capture the phase transition dynamics and generate physically realistic configurations from which
high fidelity thermodynamic observables can be calculated (like magnetisation, energy, etc.).

## Overview
This project uses **Keras** and **TensorFlow** to implement a generative model that learns the Boltzmann distribution of a 2d Ising
lattice, composed of 32x32 binary spins (0/1). Unlike a standard VAE, this CVAE is conditioned on the temperature (or inverse 
temperature, $\beta$) and also applies physical losses in training to better capture the magnetisationa and energy distributions.

## Tech Stack
- **Python 3.x**
- **Keras / Tensorflow** (Backend)
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
1. **Generate Data:** Run `python generate_data.py` to create the training lattices via Metropolis algorithm.
2. **Train:** Run `train.py` to train the model on the generated data.
3. **Tune:** Run `tune.py` for hyperparameter tuning
4. **Inference:** Use the `analysis.py` to produce graphs of the training and CVAE output behaviour.
