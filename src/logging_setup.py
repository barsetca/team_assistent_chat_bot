from __future__ import annotations

import logging
import os


def setup_logging() -> None:
    level = os.getenv("LOG_LEVEL", "INFO").upper().strip() or "INFO"
    fmt = os.getenv("LOG_FORMAT", "%(asctime)s %(levelname)s %(message)s").strip() or "%(asctime)s %(levelname)s %(message)s"
    datefmt = os.getenv("LOG_DATEFMT", None)

    logging.basicConfig(level=level, format=fmt, datefmt=datefmt)

