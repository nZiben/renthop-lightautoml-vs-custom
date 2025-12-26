from __future__ import annotations

import logging
from pathlib import Path


def setup_logging(log_path: Path | None = None, level: int = logging.INFO) -> logging.Logger:
    """Create a console/file logger with a clean format."""
    logger = logging.getLogger("renthop")
    logger.setLevel(level)
    logger.handlers.clear()

    fmt = logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s", datefmt="%H:%M:%S")

    ch = logging.StreamHandler()
    ch.setLevel(level)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_path, mode="w", encoding="utf-8")
        fh.setLevel(level)
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    logger.propagate = False
    return logger
