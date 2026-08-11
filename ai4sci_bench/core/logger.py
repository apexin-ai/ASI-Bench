"""Centralized logging configuration for ai4sci_bench."""

from __future__ import annotations

import logging
import sys


def get_logger(name: str) -> logging.Logger:
    """Get a logger with the ai4sci_bench namespace.

    Usage:
        from ai4sci_bench.core.logger import get_logger
        logger = get_logger(__name__)
        logger.info("Starting evaluation...")
    """
    return logging.getLogger(f"ai4sci_bench.{name}" if not name.startswith("ai4sci_bench") else name)


def setup_logging(level: int | str = logging.INFO, log_file: str | None = None) -> None:
    """Configure logging for the ai4sci_bench framework.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        log_file: Optional file path to write logs to (in addition to stderr).
    """
    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)

    root_logger = logging.getLogger("ai4sci_bench")
    root_logger.setLevel(level)

    # Avoid adding duplicate handlers
    if root_logger.handlers:
        return

    formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler (stderr)
    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # File handler (optional)
    if log_file:
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
