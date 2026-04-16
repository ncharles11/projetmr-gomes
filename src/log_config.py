# src/log_config.py
import logging
import sys
from src import config

def setup_logging():
    """Configures logging for the application."""
    handlers = [logging.StreamHandler(sys.stdout)]
    if config.LOG_TO_FILE:
        # Use 'a' mode to append, create file if it doesn't exist
        file_handler = logging.FileHandler(config.LOG_FILENAME, mode='a')
        handlers.append(file_handler)

    logging.basicConfig(
        level=config.LOG_LEVEL,
        format=config.LOG_FORMAT,
        handlers=handlers
    )
    # Mute overly verbose libraries if needed
    # logging.getLogger("PIL").setLevel(logging.WARNING)
    # logging.getLogger("onnxruntime").setLevel(logging.WARNING)

    logger = logging.getLogger(config.APP_NAME)
    logger.info("Logging configured.")