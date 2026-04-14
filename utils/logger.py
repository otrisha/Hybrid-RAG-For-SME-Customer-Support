"""
utils/logger.py
================
Coloured, structured logging for the Benamdaj RAG pipeline.
"""

import logging
import sys
from pathlib import Path

from config.settings import LOG_DIR

LOG_DIR.mkdir(parents=True, exist_ok=True)
_LOG_FILE = LOG_DIR / "benamdaj_rag.log"

_FMT = "%(asctime)s | %(levelname)-8s | %(name)-30s | %(message)s"
_DATE = "%Y-%m-%d %H:%M:%S"

_COLOURS = {
    "DEBUG"   : "\033[36m",   # cyan
    "INFO"    : "\033[32m",   # green
    "WARNING" : "\033[33m",   # yellow
    "ERROR"   : "\033[31m",   # red
    "CRITICAL": "\033[35m",   # magenta
    "RESET"   : "\033[0m",
}


class ColouredFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        colour = _COLOURS.get(record.levelname, "")
        reset  = _COLOURS["RESET"]
        record.levelname = f"{colour}{record.levelname}{reset}"
        return super().format(record)


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(level)

    # Console handler with colour
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(ColouredFormatter(_FMT, _DATE))
    logger.addHandler(ch)

    # File handler (plain, no colour)
    fh = logging.FileHandler(_LOG_FILE, encoding="utf-8")
    fh.setFormatter(logging.Formatter(_FMT, _DATE))
    logger.addHandler(fh)

    logger.propagate = False
    return logger
