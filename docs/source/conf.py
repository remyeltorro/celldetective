import sphinx_rtd_theme
import sys
import os
import re

#sys.path.append('/home/limozin/Documents/GitHub/rtd-tutorial')
sys.path.insert(0, os.path.abspath('./../../'))
sys.path.insert(0, os.path.abspath('./../../examples/'))

VERSIONFILE = os.path.abspath('./../../celldetective/_version.py')
verstrline = open(VERSIONFILE, "rt").read()
VSRE = r"^__version__ = ['\"]([^'\"]*)['\"]"
mo = re.search(VSRE, verstrline, re.M)
if mo:
	verstr = mo.group(1)
else:
	raise RuntimeError("Unable to find version string in %s." % (VERSIONFILE,))


# Configuration file for the Sphinx documentation builder.

# -- Project information

project = 'celldetective'
copyright = '2024, Rémy Torro'
author = 'Rémy Torro'

release = verstr
version = verstr #'1.2.2.post2'

# -- General configuration

extensions = [
	'nbsphinx',
	"sphinx.ext.autodoc",
	"sphinx.ext.intersphinx",
	"sphinx.ext.mathjax",
	"sphinx.ext.viewcode",
	'sphinx.ext.napoleon',
	'sphinx.ext.autosummary',
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

html_theme_options = {'style_nav_header_background': '#b9c3cb'}

# -- Options for EPUB output
epub_show_urls = 'footnote'
