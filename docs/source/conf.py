# Copyright 2022-2025 MetaOPT Team. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Configuration file for the Sphinx documentation builder."""
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# pylint: disable=all
# mypy: ignore-errors

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import pathlib
import sys


HERE = pathlib.Path(__file__).absolute().parent
PROJECT_ROOT = HERE.parent.parent


def get_version() -> str:
    sys.path.insert(0, str(PROJECT_ROOT / 'optree'))
    import version

    return version.__version__


# -- Project information -----------------------------------------------------

project = 'OpTree'
copyright = '2022-2025 MetaOPT Team'
author = 'OpTree Contributors'

# The full version, including alpha/beta/rc tags
release = get_version()

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.napoleon',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.doctest',
    'sphinx.ext.intersphinx',
    'sphinx.ext.todo',
    'sphinx.ext.coverage',
    'sphinx.ext.mathjax',
    'sphinx.ext.ifconfig',
    'sphinx.ext.viewcode',
    'sphinx.ext.githubpages',
    'sphinx.ext.extlinks',
    'sphinx_copybutton',
    'sphinx_rtd_theme',
    'sphinx_autodoc_typehints',
]

if not os.getenv('READTHEDOCS', None):
    extensions.append('sphinxcontrib.spelling')

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# The suffix(es) of source filenames.
# You can specify multiple suffix as a list of string:
#
# source_suffix = ['.rst', '.md']
source_suffix = {'.rst': 'restructuredtext'}

# The master toctree document.
master_doc = 'index'

# The language for content autogenerated by Sphinx. Refer to documentation
# for a list of supported languages.
#
# This is also used if you do content translation via gettext catalogs.
# Usually you set "language" from the command line for these cases.
language = 'en'

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'build', 'Thumbs.db', '.DS_Store']
spelling_exclude_patterns = ['']
spelling_word_list_filename = ['spelling_wordlist.txt']
spelling_show_suggestions = True

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = 'default'

# A list of warning codes to suppress arbitrary warning messages.
suppress_warnings = ['config.cache']

# -- Options for autodoc -----------------------------------------------------

autodoc_default_options = {
    'member-order': 'bysource',
    'undoc-members': True,
    'special-members': True,
    'show-inheritance': True,
    'exclude-members': '__module__, __dict__, __repr__, __str__, __weakref__',
}
autoclass_content = 'both'

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'

# Theme options are theme-specific and customize the look and feel of a theme
# further.  For a list of options available for each theme, see the
# documentation.
#
# html_theme_options = {}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']
html_css_files = ['style.css']
# html_logo = '_static/images/logo.png'

# Custom sidebar templates, must be a dictionary that maps document names
# to template names.
#
# The default sidebars (for documents that don't match any pattern) are
# defined by theme itself.  Builtin themes are using these templates by
# default: ``['localtoc.html', 'relations.html', 'sourcelink.html',
# 'searchbox.html']``.
#
# html_sidebars = {}

# -- Source code links -------------------------------------------------------

extlinks = {
    'gitcode': ('https://github.com/metaopt/optree/blob/HEAD/%s', '%s'),
    'issue': ('https://github.com/metaopt/optree/issues/%s', 'issue %s'),
}

# -- Extension configuration -------------------------------------------------

# -- Options for napoleon extension ------------------------------------------

napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True

# -- Options for intersphinx extension ---------------------------------------

# Example configuration for intersphinx: refer to the Python standard library.
intersphinx_mapping = {'python': ('https://docs.python.org/3', None)}

# -- Options for todo extension ----------------------------------------------

# If true, `todo` and `todoList` produce output, else they produce nothing.
todo_include_todos = True

# -- Options for copybutton extension ----------------------------------------

# To make sphinx-copybutton skip all prompt characters generated by pygments
copybutton_exclude = '.linenos, .gp'

# -- Options for autodoc-typehints extension ---------------------------------
always_use_bars_union = True
typehints_use_signature = False
typehints_use_signature_return = False


def typehints_formatter(annotation, config=None):
    from typing import Union

    if (
        isinstance(annotation, type(Union[int, str]))
        and annotation.__origin__ is Union
        and hasattr(annotation, '__pytree_args__')
    ):
        param, name = annotation.__pytree_args__
        if name is not None:
            return f':py:class:`{name}`'

        from sphinx_autodoc_typehints import format_annotation

        return rf':py:class:`PyTree` \[{format_annotation(param, config=config)}]'
    return None
