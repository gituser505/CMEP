# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os, sys

# 1. FORCE CPU AND SILENCE TENSORFLOW BEFORE ANYTHING LOADS
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
os.environ["SPHINX_BUILD"] = "1"

sys.path.insert(0, os.path.abspath('../../src'))

import sphinx_rtd_theme

project = 'CMEP project'
copyright = '2026, John Smith'
author = 'John Smith'
release = '1.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [  'sphinx.ext.autodoc',    # Extracts docstrings automatically
                'sphinx.ext.napoleon',   # Supports Google and NumPy style docstrings
                'sphinx.ext.viewcode',   # Optional: add links to source code in generated docs
                'sphinx.ext.mathjax', #
    ]

templates_path = ['_templates']
exclude_patterns = []

# Mock external dependencies to prevent import errors during build
autodoc_mock_imports = [
    'optuna'
]

language = 'y'

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
