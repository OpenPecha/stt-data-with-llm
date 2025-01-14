import logging
import os

from dotenv import load_dotenv
from google import genai

from stt_data_with_llm.util import setup_logging

load_dotenv()
setup_logging("llm_corrector.log")


# Initialize the Gemini API client
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))


def get_LLM_corrected_text(inference_text, is_valid, reference_text=None):
    """
    Corrects colloquial text with spelling mistakes using Claude API by referencing a literal sentence.

    Args:
        inference_text (str): The colloquial text with potential spelling mistakes
        reference_text (str): The literal reference text with correct spelling

    Returns:
        str: Corrected text, or None if API call fails
    """

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
            I have a colloquial sentence that may contain spelling mistakes.
            Please correct any spelling mistakes while preserving the meaning and colloquial nature of the text.
            Only fix spelling errors - do not change the style, word choice, or grammar.
            Sentence: {inference_text}
            Give me only the corrected sentence without any explanation
            """

    try:
        # Make API call to Gemini
        response = client.models.generate_content(
            model="gemini-2.0-flash-exp",
            contents=prompt,
        )
        # Extract and return the corrected text
        corrected_text = response.text.strip()

        # Extract and return the corrected text
        logging.info(
            f"Inference_transcript: {inference_text}\nReference_transcript: {reference_text}\nCorrected_text: {corrected_text}"  # noqa
        )
        return corrected_text

    except Exception as e:
        # Log error and return None if API call fails
        logging.error(f"Error in LLM correction: {str(e)}")
        return None


if __name__ == "__main__":
    inference_text = "ལྷག་པར་དགན་སྡེ་ཁག་ཏུ་ཆོས་ཕྱོགས་ཀྱི་བྱེད་སྒོ་ལ་དམ་སྒྲགས་དང་།"
    get_LLM_corrected_text(inference_text, False)
