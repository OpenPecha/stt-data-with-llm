import logging
from logging.handlers import RotatingFileHandler

from evaluate import load

from stt_data_with_llm.config import BACKUP_COUNT, MAX_BYTES

# Load the CER metric from the "evaluate" library
cer_metric = load("cer")


# Configure logging
def setup_logging(filename):
    """This function sets up a logger with both file and console handlers.

    Args:
        filename (str): The name of the log file to be created or appended to.
    """
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


def calculate_cer(reference, prediction):
    """Calculate the Character Error Rate (CER) using the evaluate library.
    args:
        reference(str): reference_transcript
        prediction(str): inference_transcript
    Returns:
        float: The calculated CER value, bounded between 0.0 and 1.0.
    """
    try:
        cer = cer_metric.compute(references=[reference], predictions=[prediction])
        return min(cer, 1.0)  # Ensure CER does not exceed 1.0
    except Exception as e:
        print(f"Error calculating CER: {e}")
        return 1.0  # Return a high CER for safety
