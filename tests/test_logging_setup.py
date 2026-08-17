"""
Tests for the shared logger factory.

The original modules built a ``logging.FileHandler`` at import time pointing
into a ``logs/`` directory that is gitignored and therefore absent from a fresh
clone. Importing ``models`` or ``preprocessing`` raised FileNotFoundError
before a single line of user code ran, and the README documented ``mkdir logs``
as a troubleshooting step rather than fixing it.

Author: BBG-Credit-Momentum
License: MIT
"""

import logging
import pathlib
import tempfile
import unittest

from logging_setup import get_logger


class TestGetLogger(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = pathlib.Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_creates_the_log_directory_when_it_is_missing(self):
        log_dir = self.root / "logs"
        self.assertFalse(log_dir.exists())

        get_logger("test_creates", "sample.log", log_dir=log_dir)

        self.assertTrue(log_dir.is_dir())

    def test_writes_records_to_the_named_file(self):
        log_dir = self.root / "logs"

        logger = get_logger("test_writes", "sample.log", log_dir=log_dir)
        logger.info("hello from the test")
        for handler in logger.handlers:
            handler.flush()

        contents = (log_dir / "sample.log").read_text()
        self.assertIn("hello from the test", contents)

    def test_repeated_calls_do_not_stack_duplicate_handlers(self):
        log_dir = self.root / "logs"

        first = get_logger("test_idempotent", "sample.log", log_dir=log_dir)
        second = get_logger("test_idempotent", "sample.log", log_dir=log_dir)

        self.assertIs(first, second)
        self.assertEqual(len(first.handlers), 1)

    def test_an_unwritable_directory_does_not_crash_the_import(self):
        # Logging is a convenience, never a reason for the application to die.
        logger = get_logger(
            "test_unwritable", "sample.log", log_dir=pathlib.Path("/proc/nope")
        )

        self.assertIsInstance(logger, logging.Logger)
        logger.info("this must not raise")


if __name__ == "__main__":
    unittest.main()
