# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os, sys
import sphinx_rtd_theme

sys.path.insert(0, os.path.abspath('../../src'))

project = 'CMEP project'
copyright = '2026, John Smith'
author = 'John Smith'
release = '1.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [    'sphinx.ext.autodoc',    # Extracts docstrings automatically
    'sphinx.ext.napoleon',   # Supports Google and NumPy style docstrings
    'sphinx.ext.viewcode',   # Optional: add links to source code in generated docs
    ]

templates_path = ['_templates']
exclude_patterns = []

# Mock external dependencies to prevent import errors during build
autodoc_mock_imports = ["tensorflow", "torch", "keras", "keras.src"]

language = 'y'

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'alabaster'
html_static_path = ['_static']
