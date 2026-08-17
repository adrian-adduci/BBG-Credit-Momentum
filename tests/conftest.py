"""
Put the repository root on sys.path.

The project modules live at the top level rather than in an installed package,
so without this the suite only collects when pytest happens to be invoked as
``python -m pytest`` from the repository root.
"""

import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).parent.parent.absolute()

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture(autouse=True, scope="session")
def _redirect_plot_output(tmp_path_factory):
    """Send every chart written during the suite to a temporary directory.

    MomentumModel accepts a plot_dir, but a test that forgets to pass one would
    silently fall back to the tracked <repo>/_img/ folder and overwrite
    committed assets with charts rendered from fixture data. Overriding the
    default for the whole session makes that impossible rather than merely
    discouraged, so `pytest` can never leave the working tree dirty.
    """
    import models

    original = models.DEFAULT_PLOT_DIR
    models.DEFAULT_PLOT_DIR = tmp_path_factory.mktemp("plots")
    try:
        yield models.DEFAULT_PLOT_DIR
    finally:
        models.DEFAULT_PLOT_DIR = original
