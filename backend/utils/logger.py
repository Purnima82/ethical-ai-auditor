"""
logger.py — Structured JSON-friendly logger for the Ethical AI Auditor.
Uses Python's built-in logging with a clean formatter.
Provides a module-level `get_logger(name)` factory.
"""
from __future__ import annotations

import logging
import sys
from typing import Optional

_RESET  = "\x1b[0m"
_COLORS = {
    "DEBUG":    "\x1b[36m",   # cyan
    "INFO":     "\x1b[32m",   # green
    "WARNING":  "\x1b[33m",   # yellow
    "ERROR":    "\x1b[31m",   # red
    "CRITICAL": "\x1b[35m",   # magenta
}


class ColourFormatter(logging.Formatter):
    """Coloured console formatter — only active when stdout is a TTY."""

    FMT = "%(asctime)s │ %(levelname)-8s │ %(name)s │ %(message)s"
    DATE_FMT = "%H:%M:%S"

    def __init__(self, use_colour: bool = True):
        super().__init__(fmt=self.FMT, datefmt=self.DATE_FMT)
        self._use_colour = use_colour and sys.stdout.isatty()

    def format(self, record: logging.LogRecord) -> str:
        msg = super().format(record)
        if self._use_colour:
            colour = _COLORS.get(record.levelname, _RESET)
            msg = f"{colour}{msg}{_RESET}"
        return msg


def get_logger(name: str, level: Optional[str] = None) -> logging.Logger:
    """
    Factory that returns a consistently configured logger.

    Args:
        name:  Logger name — use __name__ in each module.
        level: Override log level (DEBUG/INFO/WARNING/ERROR).

    Returns:
        logging.Logger
    """
    logger = logging.getLogger(name)

    if logger.handlers:
        # Already configured — avoid duplicate handlers
        return logger

    # Resolve level
    from backend.utils.config import get_settings
    try:
        settings = get_settings()
        log_level = level or settings.log_level
    except Exception:
        log_level = level or "INFO"

    logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(ColourFormatter())
    logger.addHandler(handler)
    logger.propagate = False

    return logger


# Module-level default logger
log = get_logger("ethical_ai_auditor")
