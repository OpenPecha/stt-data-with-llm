import logging
import random
import time
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


def generate_unique_id():
    timestamp = int(time.time() * 1000)  # Current time in milliseconds
    random_num = random.randint(1000, 9999)  # Random 4-digit number
    return f"{timestamp:010d}_{random_num:04d}"  # noqa: E231
