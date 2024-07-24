# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "Robpy"
copyright = "2024, STAN-Uantwerp"
author = "STAN-Uantwerp"
release = "0"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

import os
import sys

sys.path.insert(0, os.path.abspath("../.."))

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.viewcode",
    "sphinx.ext.napoleon",
    "sphinx.ext.mathjax",
]

templates_path = ["_templates"]
exclude_patterns = []

autoclass_content = "both"
autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "inherited-members": False,  # Set to False to exclude inherited members
    "show-inheritance": True,  # Optionally show the inheritance chain
}

napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True

mathjax3_config = {
    "tex": {
        "inlineMath": [["$", "$"], ["\\(", "\\)"]],
        "displayMath": [["$$", "$$"], ["\\[", "\\]"]],
        "processEscapes": True,  # Ensure backslashes are handled correctly
    }
}


def skip_sklearn_members(app, what, name, obj, skip, options):
    if hasattr(obj, "__module__") and obj.__module__.startswith("sklearn"):
        return True
    return skip


def setup(app):
    app.connect("autodoc-skip-member", skip_sklearn_members)


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
