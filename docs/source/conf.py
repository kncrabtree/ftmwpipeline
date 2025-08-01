"""
Configuration file for Sphinx documentation builder.

This file contains the configuration for generating documentation
for the ftmwpipeline package.
"""

import os
import sys
from pathlib import Path

# Add the source directory to the Python path
sys.path.insert(0, os.path.abspath('../../src'))

# Project information
project = 'ftmwpipeline'
copyright = '2025, FTMW Pipeline Contributors'
author = 'FTMW Pipeline Contributors'

# The full version, including alpha/beta/rc tags
release = '0.1.0'
version = '0.1.0'

# Extensions
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.viewcode',
    'sphinx.ext.napoleon',
    'sphinx.ext.intersphinx',
    'sphinx.ext.mathjax',
    'sphinx_rtd_theme',
]

# Napoleon settings for Google/NumPy style docstrings
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_preprocess_types = False
napoleon_type_aliases = None
napoleon_attr_annotations = True

# Autodoc settings
autodoc_default_options = {
    'members': True,
    'member-order': 'bysource',
    'special-members': '__init__',
    'undoc-members': True,
    'exclude-members': '__weakref__'
}

# Autosummary settings
autosummary_generate = True

# Add any paths that contain templates here, relative to this directory
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files
exclude_patterns = []

# The theme to use for HTML and HTML Help pages
html_theme = 'sphinx_rtd_theme'

# Theme options are theme-specific and customize the look and feel of a theme
html_theme_options = {
    'canonical_url': '',
    'analytics_id': '',
    'logo_only': False,
    'display_version': True,
    'prev_next_buttons_location': 'bottom',
    'style_external_links': False,
    'vcs_pageview_mode': '',
    'style_nav_header_background': 'white',
    'collapse_navigation': True,
    'sticky_navigation': True,
    'navigation_depth': 4,
    'includehidden': True,
    'titles_only': False
}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css"
html_static_path = ['_static']

# Custom sidebar templates, must be a dictionary that maps document names
# to template names
html_sidebars = {}

# Intersphinx mapping
intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'scipy': ('https://docs.scipy.org/doc/scipy/', None),
    'matplotlib': ('https://matplotlib.org/stable/', None),
    'pandas': ('https://pandas.pydata.org/docs/', None),
}

# Output file base name for HTML help builder
htmlhelp_basename = 'ftmwpipelinedoc'

# LaTeX configuration
latex_elements = {
    'papersize': 'letterpaper',
    'pointsize': '10pt',
    'preamble': '',
    'fncychap': '',
    'printindex': '',
}

# Grouping the document tree into LaTeX files
latex_documents = [
    ('index', 'ftmwpipeline.tex', 'ftmwpipeline Documentation',
     'FTMW Pipeline Contributors', 'manual'),
]

# Manual page output
man_pages = [
    ('index', 'ftmwpipeline', 'ftmwpipeline Documentation',
     [author], 1)
]

# Texinfo output
texinfo_documents = [
    ('index', 'ftmwpipeline', 'ftmwpipeline Documentation',
     author, 'ftmwpipeline', 'FTMW spectroscopy signal processing and peak fitting.',
     'Miscellaneous'),
]

# Epub output
epub_title = project
epub_exclude_files = ['search.html']