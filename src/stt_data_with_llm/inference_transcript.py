import logging
import wave
from io import BytesIO

import numpy as np
from transformers import pipeline


def convert_raw_to_wav_in_memory(
    raw_audio, sample_rate=16000, channels=1, sample_width=2
):
    """
    Converts raw audio data to a valid WAV format in memory.

    Args:
        raw_audio (bytes): Raw audio data.
        sample_rate (int): Audio sample rate (default is 16000 Hz).
        channels (int): Number of audio channels (default is 1 for mono).
        sample_width (int): Number of bytes per sample (default is 2 for 16-bit audio).

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


def wav_to_numpy(wav_buffer):
    """
    Converts an in-memory WAV file to a numpy array.

    Args:
        wav_buffer (BytesIO): In-memory WAV file buffer.

    Returns:
        np.ndarray: Audio data as a numpy array.
        int: Sample rate of the audio.
    """
    try:
        wav_buffer.seek(0)
        with wave.open(wav_buffer, "rb") as wav_file:
            frames = wav_file.readframes(wav_file.getnframes())
            sample_rate = wav_file.getframerate()
            channels = wav_file.getnchannels()

            # Convert raw frames to numpy array
            audio_data = np.frombuffer(frames, dtype=np.int16)  # Assuming 16-bit PCM
            if channels > 1:
                audio_data = audio_data.reshape(-1, channels)
            return audio_data, sample_rate
    except Exception as e:
        logging.error(f"Error converting WAV to numpy array: {e}")
        return None, None


def get_audio_inference_text(raw_audio):
    """
    Generates the inference transcript for raw audio data.

    Args:
        raw_audio (bytes): Raw audio data of the segment.

    Returns:
        str: The transcript generated for the given audio segment.
    """
    try:
        # Define the model pipeline for inference
        generator = pipeline(
            task="automatic-speech-recognition", model="spsither/mms_300_v3.1020"
        )

        # Convert raw audio to WAV format in memory
        wav_buffer = convert_raw_to_wav_in_memory(raw_audio)
        if not wav_buffer:
            return ""

        # Convert WAV to numpy array
        audio_data, _ = wav_to_numpy(wav_buffer)  # Only pass the audio data
        if audio_data is None:
            return ""

        logging.info("Running inference on audio segment")

        # Generate the inference text using the numpy array
        result = generator(audio_data)
        transcript = result["text"]

        logging.info("Inference completed successfully")
        return transcript

    except Exception as e:
        logging.error(f"Error during inference: {e}")
        return ""
