# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os, re
import sys, shutil
from glob import glob
from pathlib import Path
from git.repo import Repo

DIR = Path(__file__).parent.resolve()
ROOT = DIR.parent

sys.path.insert(0, str(ROOT))


def download_image():
    static_path = DIR / '_static/cigvis'
    if static_path.exists():
        print('Use existing gallery images')
        return

    print('Run git to download images')
    down_path = DIR / '_static/images'
    if down_path.exists():
        shutil.rmtree(down_path)
    Repo.clone_from('https://github.com/JintaoLee-Roger/images.git',
                    to_path=down_path)

    shutil.move(down_path / 'cigvis', down_path.parent)
    shutil.rmtree(down_path)


download_image()

from sphinx_gallery.scrapers import figure_rst


class PNGScraper(object):

    def __init__(self):
        self.seen = set()

    def __repr__(self):
        return 'PNGScraper'

    def __call__(self, block, block_vars, gallery_conf):
        # Find all PNG files in the directory of this example.
        path_current_example = os.path.dirname(block_vars['src_file'])
        pngs = sorted(glob(os.path.join(path_current_example, '*.png')))

        # Iterate through PNGs, copy them to the sphinx-gallery output directory
        image_names = list()
        image_path_iterator = block_vars['image_path_iterator']
        for png in pngs:
            if png not in self.seen:
                self.seen |= set(png)
                this_image_path = image_path_iterator.next()
                image_names.append(this_image_path)
                shutil.move(png, this_image_path)
        # Use the `figure_rst` helper function to generate reST for image files
        return figure_rst(image_names, gallery_conf['src_dir'])


project = 'cigvis'
copyright = '2024, Jintao Li'
author = 'Jintao Li and others'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinxcontrib.apidoc',
    'sphinx_gallery.gen_gallery',
    'myst_parser',
]

apidoc_module_dir = "../cigvis"
apidoc_output_dir = "api"
apidoc_excluded_paths = ["../cigvis/cpp"]
apidoc_separate_modules = True

# """
# Sphinx Gallery
from sphinx_gallery.sorting import FileNameSortKey

# the following files are ignored from gallery processing
ignore_files = [
    r'demos/.*',
    r'test.*\.py',
]
ignore_pattern_regex = [r'(^|.*/)' + f for f in ignore_files]
ignore_pattern_regex = "|".join(ignore_pattern_regex)

execute = False
sphinx_gallery_conf = {
    'examples_dirs': [
        '../examples/3Dvispy', '../examples/2D', '../examples/1D',
        '../examples/colormap', '../examples/more_demos', '../examples/viser',
        '../examples/sliceviewer'
    ],
    'gallery_dirs': [
        'gallery/3Dvispy', 'gallery/2D', 'gallery/1D', 'gallery/colormap',
        'gallery/more_demos', 'gallery/viser', 'gallery/sliceviewer'
    ],
    'filename_pattern':
    re.escape(os.sep),
    'ignore_pattern':
    ignore_pattern_regex,
    'only_warn_on_example_error':
    True,
    'image_scrapers': (PNGScraper(), ),
    'reset_modules':
    tuple(),  # remove default matplotlib/seaborn resetters
    'first_notebook_cell':
    '%gui qt',  # tell notebooks to use Qt backend
    'within_subsection_order':
    FileNameSortKey,
    'download_all_examples':
    False,
}
if not execute:
    sphinx_gallery_conf['plot_gallery'] = False

# Let vispy.app.application:Application.run know that we are generating gallery images
os.environ["_VISPY_RUNNING_GALLERY_EXAMPLES"] = "1"

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store', "**.ipynb_checkpoints"]
suppress_warnings = ['config.cache']
# """

# ReadTheDocs has its own way of generating sitemaps, etc.
# if not os.environ.get("READTHEDOCS"):
#     extensions += ["sphinx_sitemap"]

#     html_baseurl = os.environ.get("SITEMAP_URL_BASE", "http://127.0.0.1:8000/")
#     sitemap_locales = [None]
#     sitemap_url_scheme = "{link}"

language = 'zh_CN'

# The suffix of source filenames.
source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
}

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'pydata_sphinx_theme'
html_static_path = ['_static']

# The main toctree document.
master_doc = 'index'

html_theme_options = {
    "use_edit_page_button":
    True,
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/JintaoLee-Roger/cigvis",
            "icon": "fa-brands fa-github",
        },
        {
            "name": "PyPI",
            "url": "https://pypi.org/project/cigvis/",
            "icon": "fa-custom fa-pypi",
        },
    ]
}

html_context = {
    "github_url":
    "https://github.com/JintaoLee-Roger/cigvis",  # or your GitHub Enterprise site
    "github_user": "JintaoLee-Roger",
    "github_repo": "cigvis",
    "github_version": "main",
    "doc_path": "https://cigvis.readthedocs.io/en/latest/",
}

html_title = 'CIGVis'

html_show_sourcelink = True
html_show_sphinx = True
html_show_copyright = True

htmlhelp_basename = 'cigvisdoc'


def _patch_generated_api_docs(app):
    """Tweak apidoc output that is regenerated on every Sphinx build."""
    gui_api = DIR / 'api/cigvis.gui.rst'
    if gui_api.exists():
        text = gui_api.read_text()
        marker = ".. automodule:: cigvis.gui\n"
        replacement = ".. automodule:: cigvis.gui\n   :no-index:\n"
        if marker in text and replacement not in text:
            gui_api.write_text(text.replace(marker, replacement))


def setup(app):
    app.connect('builder-inited', _patch_generated_api_docs, priority=900)
