import logging
import os
import wave
from io import BytesIO

import requests
from dotenv import load_dotenv

from stt_data_with_llm.config import API_URL, CHANNELS, SAMPLE_RATE, SAMPLE_WIDTH
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


def convert_raw_to_wav_in_memory(raw_audio, sample_rate, channels, sample_width):
    """
    Converts raw audio data to a valid WAV format in memory.

    Args:
        raw_audio (bytes): Raw audio data.
        sample_rate (int): Audio sample rate.
        channels (int): Number of audio channels.
        sample_width (int): Number of bytes per sample.

    Returns:
        BytesIO: In-memory WAV file if conversion is successful, None otherwise.
    """
    try:
        wav_buffer = BytesIO()
        with wave.open(wav_buffer, "wb") as wav_file:
            wav_file.setnchannels(channels)
            wav_file.setsampwidth(sample_width)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(raw_audio)
        wav_buffer.seek(0)  # Reset buffer to the beginning
        logging.info("Raw audio successfully converted to WAV format in memory.")
        return wav_buffer
    except Exception as e:
        logging.error(f"Error converting raw audio to WAV in memory: {e}")
        return None


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


def get_audio_inference_text(raw_audio):
    """
    Generates the inference transcript for raw audio data.

    Args:
        raw_audio (bytes): Raw audio data of the segment.

    Returns:
        str: The transcript generated for the given audio segment.
    """
    try:
        # Convert raw audio to WAV format in memory
        wav_buffer = convert_raw_to_wav_in_memory(
            raw_audio, SAMPLE_RATE, CHANNELS, SAMPLE_WIDTH
        )
        if not wav_buffer:
            return ""
        logging.info("Running inference on audio segment")
        # Send the WAV data to the API for transcription
        response = query_audio_api(wav_buffer)
        if not response or "text" not in response:
            return ""
        transcript = response["text"]

        logging.info("Inference completed successfully")
        return transcript

    except Exception as e:
        logging.error(f"Error during inference: {e}")
        return ""
