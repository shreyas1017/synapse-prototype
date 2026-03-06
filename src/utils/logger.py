"""
Centralized logging for SYNAPSE system.
Logs to both console (INFO+) and file (DEBUG+).
"""

import logging
import os
from datetime import datetime


def setup_logger(name: str = "SYNAPSE", log_dir: str = "logs") -> logging.Logger:
    """
    Set up logger with file + console handlers.

    Args:
        name: Logger name
        log_dir: Directory to store log files

    Returns:
        Configured logger instance
    """
    # Create logs directory
    os.makedirs(log_dir, exist_ok=True)

    # Log filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"synapse_{timestamp}.log")

    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # Avoid duplicate handlers on re-init
    if logger.handlers:
        return logger

    # --- File handler (DEBUG level - captures everything) ---
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    file_handler.setFormatter(file_formatter)

    # --- Console handler (INFO level - only important messages) ---
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter("%(message)s")
    console_handler.setFormatter(console_formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    logger.info(f"[LOGGER] Logging to: {log_file}")

    return logger


# Global logger instance
logger = setup_logger()
