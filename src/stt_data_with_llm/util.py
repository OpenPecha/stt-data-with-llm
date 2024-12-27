import logging
from logging.handlers import RotatingFileHandler

from stt_data_with_llm.config import BACKUP_COUNT, MAX_BYTES


# Configure logging
def setup_logging(filename):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Create a formatter
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    # Create a file handler for a rotating log file
    file_handler = RotatingFileHandler(
        filename,
        MAX_BYTES,
        BACKUP_COUNT,
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Create a console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
