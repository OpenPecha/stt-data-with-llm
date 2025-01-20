import logging
import os

import anthropic
from dotenv import load_dotenv

load_dotenv()


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
            Your task is to correct Colloquial Tibetan sentences by comparing them with Reference sentences. Here are your specific responsibilities:
            Main Task:
            - Correct spelling and grammar mistakes in the Colloquial sentence by carefully comparing with the Reference sentence
            - Preserve All Chinese-derived terms (regardless of their spelling),The original structure of compound terms basic terms and sentence structure unless there's a clear mistake
            - Convert written numbers to Tibetan numerals (e.g., བདུན་ཅུ་ to ༧༠)
            - DO NOT add or remove particles (like དུ་, ནི་, etc.) that aren't present in the original colloquial text
            - DO NOT modify word order or syntax from the original colloquial text
            - Particles that appear differently in the Reference sentence (e.g., ཀྱི་ should be ཀྱིས if Reference shows ཀྱིས)


            Guidelines:
            1. DO CHANGE:
            - Incorrect spellings of pure Tibetan words (e.g., ཚོས་ should be ཆོས་)
            - Basic Tibetan particle spelling mistakes (like ཀི་ to ཀྱི་)
            - Particles that appear differently in the Reference sentence (e.g., ཀྱི་ should be ཀྱིས if Reference shows ཀྱིས)
            - Obviously incorrect syllable formation in pure Tibetan words
            - Number formats (convert to Tibetan numerals)
            - Clear grammatical errors in Tibetan particles
            - Words that appear in the Reference sentence with different spelling (e.g., ཐབས་ should be འཐབ་ if Reference shows འཐབ་)
            2. DO NOT CHANGE:
            - Any Chinese words written in Tibetan script (MOST IMPORTANT)
            - ANY term that might be derived from Chinese
            - DO NOT add or remove particles (like དུ་, ནི་, etc.) that aren't present in the original colloquial text


            3. PRESERVE EXACTLY:
            - All Chinese-derived terms (regardless of their spelling)
            - The original structure of compound terms

            Example:
            Colloquial: ཚོས་ལུགས་ལས་དོན་གཅོས་ཀྱི་ཐབས་གཅོག་གཅིག་ྒྱུར་ལས་དོན་འབྲེལ་ཡོད་ལས་བྱེད་དགོན་པ་ཁག་ཏུ་ཉུས་ཞིབ་ཀི་དགོན་སྡེ་ཁག
            Reference: འཐབ་ཕྱོགས་གཅིག་གྱུར་ལས་དོན་དང་འབྲེལ་ཡོད་ལས་བྱེད་དགོན་པ་ཁག་ཏུ་ཉུལ་ཞིབ་ཀྱིས
            Corrected: ཆོས་ལུགས་ལས་དོན་གཅོས་ཀྱི་འཐབ་གཅོག་གཅིག་གྱུར་ལས་དོན་འབྲེལ་ཡོད་ལས་བྱེད་དགོན་པ་ཁག་ཏུ་ཉུལ་ཞིབ་ཀྱི་དགོན་སྡེ་ཁག

            Format:
            Colloquial: {inference_text}
            Reference: {reference_text}

            Note:
            - Output ONLY the corrected sentence with no additional text or explanations
            - If you notice spelling mistakes in the Reference sentence, rely on standard Tibetan orthography rather than the Reference sentence
            - Don't add or delete words from the Colloquial sentence unless there's a clear mistake
            - Cross-reference spellings with the Reference sentence when available
            - Use the Reference sentence for both spelling verification and correct word forms
            """  # noqa
    else:
        prompt = f"""
            You are a Tibetan Language Expert specializing in modern Chinese-Tibetan terminology. Analyze the following Tibetan sentence with these STRICT requirements:

            1. DO CHANGE:
            - Incorrect spellings of pure Tibetan words (e.g., ཚོས་ should be ཆོས་)
            - Basic Tibetan particle spelling mistakes (like ཀི་ to ཀྱི་)
            - Obviously incorrect syllable formation in pure Tibetan words
            - Number formats (convert to Tibetan numerals)
            - Clear grammatical errors in Tibetan particles

            2. DO NOT CHANGE:
            - Any Chinese words written in Tibetan script (MOST IMPORTANT)
            - ANY term that might be derived from Chinese

            3. PRESERVE EXACTLY:
            - All Chinese-derived terms (regardless of their spelling)
            - The original structure of compound terms

            Tibetan sentence: {inference_text}

            Output: Return only the corrected sentence without any explanation or additional
            CRITICAL:
            - Preserve ALL Chinese-derived terms exactly as written
            - DO correct misspelled pure Tibetan words (like ཚོས་ to ཆོས་)
            - Only correct obvious Tibetan grammar particles and numbering

            Remember: When in doubt about whether a term is Chinese-derived, preserve the original spelling.
      """  # noqa
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
