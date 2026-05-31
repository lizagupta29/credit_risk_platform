"""
logger.py
"""

import logging
import os


def setup_logger(
    name="credit_risk_app",
    log_file="app.log",
    level=logging.INFO
):

    logger = logging.getLogger(name)

    logger.setLevel(level)

    if not logger.handlers:

        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)s | %(message)s"
        )

        file_handler = logging.FileHandler(
            log_file
        )

        file_handler.setFormatter(
            formatter
        )

        stream_handler = logging.StreamHandler()

        stream_handler.setFormatter(
            formatter
        )

        logger.addHandler(
            file_handler
        )

        logger.addHandler(
            stream_handler
        )

    return logger


logger = setup_logger()
