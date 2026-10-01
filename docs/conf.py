import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

project = "cgrid_tools"
author = "cgrid_tools contributors"
release = __import__("cgrid_tools").__version__

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "myst_nb",
    "sphinx_autodoc_typehints",
]

# myst-nb execution settings
nb_execution_mode = "auto"  # execute notebooks if outputs are missing/stale
nb_execution_timeout = 600  # allow up to 10 min per notebook
nb_execution_raise_on_error = True  # fail build on notebook error

html_theme = "furo"
autosummary_generate = True
autodoc_typehints = "description"
napoleon_numpy_docstring = True

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "scipy": ("https://docs.scipy.org/doc/scipy", None),
    "xarray": ("https://docs.xarray.dev/en/stable", None),
}

source_suffix = {".rst": "restructuredtext", ".md": "myst-nb", ".ipynb": "myst-nb"}

exclude_patterns = ["_build", "_build/**", "Thumbs.db", ".DS_Store"]
