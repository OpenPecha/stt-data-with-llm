import logging
import os

import anthropic
from dotenv import load_dotenv

from stt_data_with_llm.util import setup_logging

load_dotenv()
setup_logging("llm_corrector.log")


def get_LLM_corrected_text(inference_text, is_valid, reference_text=None):
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

    if is_valid and reference_text is not None:
        prompt = f"""
            I have two sentences: a colloquial sentence and a reference sentence.
            Your task is to EXACTLY match the spellings from the reference sentence.
            Do not make any corrections beyond matching the reference sentence exactly, even if you think a word is misspelled.   # noqa
            If a word appears the same way in both sentences, do not change it.
            Colloquial sentence: {inference_text}
            Reference sentence: {reference_text}
            Give me only the corrected sentence that exactly matches the reference, without any explanation
            """
    else:
        prompt = f"""
            You are a Tibetan Language Expert. I want you to look for any spelling and grammar mistakes in the following Tibetan
            sentence. Make sure that you don't change the terms and sentence if its not grammatically incorrect.
            Tibetan sentence: {inference_text}
            Output: output should be only the corrected sentence.
            Give me only the corrected sentence without any explanation
            """

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
