################################################################################
# Author: Adrian Adduci
# Email: FAA2160@columbia.edu
################################################################################
"""
Shared logger factory.

Modules used to attach a ``logging.FileHandler`` at import time pointing into a
gitignored ``logs/`` directory, so a fresh clone raised FileNotFoundError on
``import _models`` before any user code ran.

``get_logger`` creates the directory on demand and degrades to a null handler
if the filesystem refuses. Logging is a diagnostic aid; it is never a reason
for the application to fail to start.
"""

import logging
import pathlib
from typing import Optional, Union

__all__ = ["get_logger", "DEFAULT_LOG_DIR"]

DEFAULT_LOG_DIR = pathlib.Path(__file__).parent.absolute() / "logs"

_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"


def get_logger(
    name: str,
    filename: str,
    log_dir: Optional[Union[str, pathlib.Path]] = None,
    level: int = logging.INFO,
) -> logging.Logger:
    """
    Return a named logger writing to ``log_dir/filename``.

    Safe to call repeatedly: a logger that already has handlers is returned
    unchanged rather than accumulating a duplicate handler per import.

    Args:
        name: Logger name, e.g. ``"_model"``.
        filename: Log file name, e.g. ``"_model.log"``.
        log_dir: Destination directory. Defaults to ``<package>/logs``.
        level: Logging level (default: ``logging.INFO``).

    Returns:
        logging.Logger: Configured logger. If the destination cannot be
        created or opened, the logger is returned with a null handler so that
        callers never have to guard their logging calls.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if logger.handlers:
        return logger

    directory = pathlib.Path(log_dir) if log_dir is not None else DEFAULT_LOG_DIR

    try:
        directory.mkdir(parents=True, exist_ok=True)
        handler: logging.Handler = logging.FileHandler(directory / filename)
        handler.setFormatter(logging.Formatter(_FORMAT))
    except OSError:
        # Read-only filesystem, missing permissions, container without a
        # writable working directory -- none of which should stop the app.
        handler = logging.NullHandler()

    logger.addHandler(handler)
    return logger
