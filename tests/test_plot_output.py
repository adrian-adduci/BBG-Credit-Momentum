"""Tests that plotting writes where the caller asks, not into the repository.

MomentumModel wrote its charts to a hardcoded ``<repo>/_img/`` directory. That
is a tracked location, so simply running the test suite overwrote committed
repository assets and left the working tree dirty -- and the overwritten
charts were regenerated from whatever fixture data the tests happened to use.

Author: BBG-Credit-Momentum
License: MIT
"""

import pathlib
import unittest

import models


class TestPlotOutputDirectory(unittest.TestCase):
    def test_module_defines_the_repo_img_folder_as_its_default(self):
        """Application callers still get <repo>/_img by default.

        Asserted against the source rather than the live value: a session
        fixture in conftest.py redirects DEFAULT_PLOT_DIR for the duration of
        the suite, so the runtime value here is a temporary directory.
        """
        source = pathlib.Path(models.__file__).read_text()

        self.assertIn('DEFAULT_PLOT_DIR = path / "_img"', source)

    def test_default_is_a_path(self):
        self.assertIsInstance(models.DEFAULT_PLOT_DIR, pathlib.Path)

    def test_plot_dir_is_configurable(self):
        """Tests and callers can redirect chart output somewhere disposable."""
        import inspect

        params = inspect.signature(models.MomentumModel.__init__).parameters
        self.assertIn("plot_dir", params)
        self.assertIsNone(
            params["plot_dir"].default,
            "plot_dir must default to None so existing callers are unaffected",
        )

    def test_resolve_plot_dir_creates_the_target(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            target = pathlib.Path(tmp) / "charts"
            resolved = models._resolve_plot_dir(target)

            self.assertEqual(resolved, target)
            self.assertTrue(resolved.is_dir())

    def test_resolve_plot_dir_falls_back_to_the_default(self):
        self.assertEqual(models._resolve_plot_dir(None), models.DEFAULT_PLOT_DIR)


class TestNoHardcodedImgWrites(unittest.TestCase):
    def test_no_savefig_targets_a_hardcoded_img_path(self):
        """A hardcoded target at the call site is what mutated tracked files.

        A single DEFAULT_PLOT_DIR definition is fine -- it is the override
        point. What must not come back is `_img` appearing in a savefig call.
        """
        source = pathlib.Path(models.__file__).read_text().splitlines()

        offenders = [
            line.strip()
            for line in source
            if "savefig" in line and "_img" in line
        ]
        self.assertEqual(offenders, [], f"savefig writes to a fixed path: {offenders}")

    def test_default_plot_dir_is_defined_exactly_once(self):
        source = pathlib.Path(models.__file__).read_text()

        self.assertEqual(source.count('path / "_img"'), 1)
