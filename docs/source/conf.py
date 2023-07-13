import sphinx_rtd_theme
import sys
import os

#sys.path.append('/home/limozin/Documents/GitHub/rtd-tutorial')
sys.path.insert(0, os.path.abspath('../../'))
sys.path.insert(0, os.path.abspath('../../examples/'))

# Configuration file for the Sphinx documentation builder.

# -- Project information

project = 'CellDetective'
copyright = '2023, Rémy Torro'
author = 'Rémy Torro'

release = '0.0'
version = '0.0.0'

# -- General configuration

#import adccfactory.core.utils as u
#print(dir(u))



extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    'sphinx.ext.napoleon',
    'sphinx.ext.autosummary',
    'nbsphinx',
    'nbsphinx_link',
]


autoapi_dirs = ['celldetective']

intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'sphinx': ('https://www.sphinx-doc.org/en/master/', None),
}
intersphinx_disabled_domains = ['std']

templates_path = ['_templates']

# -- Options for HTML output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
html_logo = "_static/logo.png"

html_theme_options = {'style_nav_header_background': 'black'}


# -- Options for EPUB output
epub_show_urls = 'footnote'
