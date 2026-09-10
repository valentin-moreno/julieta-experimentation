"""A logging module to setup loggings."""

import logging
from pathlib import Path


def setup_logging(
    log_file_path: str | Path | None = None,
) -> None:
    """
    Sets up logging to console, local file, or both.

    Parameters
    ----------
    log_file_path : Optional[Union[str, Path]]
        The path to the local log file.
        If None, local file logging will be disabled, by default None
    """
    # Get the default root logger
    logger = logging.getLogger()

    # Capture all logging levels by the logger
    logger.setLevel(logging.DEBUG)

    # Create the same logging format for file and console
    formatter = logging.Formatter(
        fmt='{"severity": "%(levelname)s", "message": "%(message)s", "timestamp": "%(asctime)s"}',
        datefmt="%Y-%m-%dT%H:%M:%S%z",
    )

    # Clear existing handlers (if any)
    logger.handlers = []

    # Configure local file logging if log_file is provided
    if log_file_path is not None:
        file_handler = logging.FileHandler(log_file_path)
        file_handler.setLevel(logging.ERROR)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    # Add console handler for logging to console
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)


if __name__ == "_main_":
    # Setup a test logging
    setup_logging(log_file_path="log/test_error_log.txt")

    # Test the logging logging.debug("This is a debug message.")
    logging.info("This is an info message.")
    logging.warning("This is a warning message.")
    logging.error("This is an error message.")
    logging.critical("This is a critical message.")

    TEXT1 = "Dummy text1"
    TEXT2 = "Dummy text2"
    logging.info("This is an info message with 2 texts: %s, %s", TEXT1, TEXT2)
