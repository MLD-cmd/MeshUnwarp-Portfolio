# Both processes call init_logging() at startup; logs append to logs/run.log and stderr.

import logging
import os
from pathlib import Path

_LOG_PATH = Path(__file__).resolve().parent.parent / "logs" / "run.log"
_INITIALIZED = False


def init_logging(role: str = "main") -> logging.Logger:
    global _INITIALIZED
    logger = logging.getLogger()
    if _INITIALIZED:
        return logger

    _LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

    fmt = logging.Formatter(
        f"%(asctime)s [%(levelname)s] [{role}] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    file_h = logging.FileHandler(_LOG_PATH, mode="a", encoding="utf-8")
    file_h.setFormatter(fmt)

    stream_h = logging.StreamHandler()
    stream_h.setFormatter(fmt)

    logger.handlers.clear()
    logger.addHandler(file_h)
    logger.addHandler(stream_h)
    logger.setLevel(os.environ.get("MESHUNWARP_LOG_LEVEL", "INFO").upper())

    _INITIALIZED = True
    return logger
