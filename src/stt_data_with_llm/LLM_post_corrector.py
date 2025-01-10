import logging
import os
from typing import Optional

import anthropic
from dotenv import load_dotenv

from stt_data_with_llm.util import setup_logging

load_dotenv()
setup_logging("llm_corrector_log.log")


def get_LLM_corrected_text(inference_text: str, reference_text: str) -> Optional[str]:
    """
    Corrects colloquial text with spelling mistakes using Claude API by referencing a literal sentence.

    Args:
        inference_text (str): The colloquial text with potential spelling mistakes
        reference_text (str): The literal reference text with correct spelling

    Returns:
        str: Corrected text, or None if API call fails
    """
    # Initialize the Anthropic client
    client = anthropic.Client(api_key=os.getenv("ANTHROPIC_API_KEY"))

    # Construct the prompt
    prompt = (
        "I have a colloquial sentence with incorrect spelling. "
        "I have a literal sentence with correct spelling. "
        "Please refer to literal sentence and correct the spelling mistakes in colloquial one. "
        f"colloquial sentence: {inference_text} "
        f"literal sentence: {reference_text} "
        "Give me the final corrected sentence without any explanation"
    )

    try:
        # Make API call to Claude
        response = client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=1000,
            messages=[{"role": "user", "content": prompt}],
        )

        # Extract and return the corrected text
        logging.info(
            f"Inference_transcript: {inference_text}\nReference_transcript: {reference_text}\nCorrected_text: {response.content[0].text.strip()}"  # noqa
        )
        return response.content[0].text.strip()

    except Exception as e:
        # Log error and return None if API call fails
        logging.error(f"Error in LLM correction: {str(e)}")
        return None
