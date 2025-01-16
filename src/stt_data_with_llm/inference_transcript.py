import io
import logging
import os

import requests
from dotenv import load_dotenv

from stt_data_with_llm.config import API_URL
from stt_data_with_llm.util import setup_logging

load_dotenv()
TOKEN_ID = os.getenv("token_id")
# Call the setup_logging function at the beginning of your script
setup_logging("inference_log.log")

INFERENCE_HEADERS = {
    "Accept": "application/json",
    "Authorization": f"Bearer {TOKEN_ID}",
    "Content-Type": "audio/wav",
}


def query_audio_api(wav_buffer):
    """
    Sends the WAV audio data to the Hugging Face API for inference.

    Args:
        wav_buffer (BytesIO): In-memory WAV file buffer.

    Returns:
        dict: API response containing the transcription.
    """
    try:
        response = requests.post(API_URL, headers=INFERENCE_HEADERS, data=wav_buffer)
        response.raise_for_status()
        api_response = response.json()
        logging.info("API call successful")
        return api_response
    except requests.RequestException as e:
        logging.error(f"Error during API call: {e}")
        return None


def get_audio_inference_text(audio_segment):
    """
    Generates the inference transcript for raw audio data.

    Args:
        raw_audio (bytes): Raw audio data of the segment.

    Returns:
        str: The transcript generated for the given audio segment.
    """
    try:
        # Convert raw audio to WAV format in memory
        # wav_buffer = convert_raw_to_wav_in_memory(
        #     raw_audio, SAMPLE_RATE, CHANNELS, SAMPLE_WIDTH
        # )"""
        if not audio_segment:
            return ""
        logging.info("Running inference on audio segment")
        # Convert AudioSegment to WAV format in memory
        buffer = io.BytesIO()
        audio_segment.export(buffer, format="wav")
        buffer.seek(0)  # Reset buffer to the beginning

        # Send the WAV data to the API for transcription
        response = query_audio_api(buffer)
        if not response or "text" not in response:
            return ""
        transcript = response["text"]

        logging.info("Inference completed successfully")
        return transcript

    except Exception as e:
        logging.error(f"Error during inference: {e}")
        return ""
