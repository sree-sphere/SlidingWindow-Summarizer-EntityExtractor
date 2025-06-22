import logging
from logging.handlers import RotatingFileHandler
import os


def setup_logging(log_filename: str) -> logging.Logger:
    """
    Configures logging for the application.

    Args:
        log_filename: Name of the log file.
    """
    # Set up logging configuration
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Create a rotating file handler
    handler = RotatingFileHandler(
        log_filename, maxBytes=10 * 1024 * 1024, backupCount=10
    )
    handler.setLevel(logging.DEBUG)

    # Create a console handler to log messages to stdout
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.ERROR)

    # Define the format for log messages
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # Add handlers to the logger
    logger.addHandler(handler)
    logger.addHandler(console_handler)

    # Log a message indicating that logging has been configured
    logger.info("Logging is configured.")

    return logger


log_dir = "log"
log_file_path = os.path.join(log_dir, "app.log")

# Create the directory if it doesn't exist
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
    print(f"Directory '{log_dir}' created.")
else:
    print(f"Directory '{log_dir}' already exists.")

# Set up the logger
logger = setup_logging(log_filename=log_file_path)