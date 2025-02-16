"""Logger module takes charge of recording information, warnings and errors
during executing tasks."""

__version__ = "0.1.0"

import logging
import sys

logger = logging.getLogger("SDL-Orchestration")
if not logger.hasHandlers():
    logger.setLevel(
        logging.INFO)  # Set logger to capture DEBUG level messages and above.
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(
        logging.INFO)  # Set handler to pass through DEBUG level messages
    # and above.
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(funcName)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    file_handler = logging.FileHandler("sdl_orchestration.log")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
