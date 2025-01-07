import io
import logging
import os

import librosa
import requests
import torchaudio
from pyannote.audio import Pipeline
from pydub import AudioSegment

from stt_data_with_llm.config import (
    AUDIO_HEADERS,
    AUDIO_SEG_LOWER_LIMIT,
    AUDIO_SEG_UPPER_LIMIT,
    HYPER_PARAMETERS,
)
from stt_data_with_llm.util import setup_logging

# Call the setup_logging function at the beginning of your script
setup_logging("audio_parser.log")


def sec_to_millis(seconds):
    """Converts seconds to milliseconds.

    Args:
        seconds (float): Time in seconds

    Returns:
        float: Time in milliseconds
    """
    return seconds * 1000


def frame_to_sec(frame, sampling_rate):
    """Converts audio frames to seconds based on sampling rate.

    Args:
        frame (int): audio frame number
        sampling_rate (int): Audio sampling rate in Hz

    Returns:
        int: Time in seconds
    """
    return frame / sampling_rate


def sec_to_frame(sec, sr):
    """Converts seconds to frame number based on sampling rate.

    Args:
        sec (float): Time in seconds
        sr (int): Audio sampling rate in Hz

    Returns:
        float: Frame number
    """
    return sec * sr


pipeline = Pipeline.from_pretrained(
    "pyannote/voice-activity-detection",
    use_auth_token="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",
)
pipeline.instantiate(HYPER_PARAMETERS)


def save_segment(segment, folder, prefix, id, start_ms, end_ms):
    """Saves an audio segment to WAV file with standardized naming.

    Args:
        segment (AudioSegment): Audio segment to save
        folder (str): Output directory path
        prefix (str): Filename prefix
        id (int): Segment Identifier
        start_ms (float): Segment start time in milliseconds
        end_ms (float): Segment end time in milliseconds
    """
    segment.export(
        f"{folder}/{prefix}_{id:04}_{int(start_ms)}_to_{int(end_ms)}.wav",  # noqa: E231
        format="wav",
        parameters=["-ac", "1", "-ar", "16000"],
    )


def convert_to_16K(audio_data):
    """Converts audio data to 16kHz mono WAV format.

    Args:
        audio_data (bytes): Raw audio data

    Returns:
        bytes: Converted 16kHz mono audio data, if conversion fails then returns None
    """
    try:
        # Load the audio data into an AudioSegment
        audio = AudioSegment.from_file(io.BytesIO(audio_data))

        # Resample to 16kHz and set to mono
        audio_16k = audio.set_frame_rate(16000).set_channels(1)

        # Export to bytes
        output_buffer = io.BytesIO()
        audio_16k.export(output_buffer, format="wav")
        output_buffer.seek(0)  # Reset buffer pointer to the beginning

        return output_buffer.read()  # Return the 16kHz audio data as bytes
    except Exception as e:
        print(f"Error during conversion: {e}")
        return None


def get_audio(audio_url):
    """Downloads and converts audio from URL to 16kHz format.


    Args:
        audio_url (str): URL of the audio file

    Raises:
        Exception: If download fails

    Returns:
        bytes: Downloaded and converted audio data
    """
    logging.info(f"Downloading audio from: {audio_url}")
    response = requests.get(audio_url, headers=AUDIO_HEADERS, stream=True)
    if response.status_code == 200:
        audio_data = response.content  # Store original audio in memory
        logging.info("Converting Audio to 16k")
        audio_data_16k = convert_to_16K(audio_data)
        logging.info("Audio downloaded and converted to 16kHz successfully")
        return audio_data_16k
    else:
        err_message = f"Failed to download audio from {audio_url}"
        logging.error(err_message)
        raise Exception(err_message)


def get_split_audio(
    audio_data,
    full_audio_id,
    lower_limit=AUDIO_SEG_LOWER_LIMIT,
    upper_limit=AUDIO_SEG_UPPER_LIMIT,
):
    """Splits audio into segments based on voice activity detection.

    Args:
        audio_data (bytes): Raw audio data
        lower_limit (float): Minimum segment duration in seconds
        upper_limit (_type_): Maximum segment duration in seconds
        full_audio_id (str):  Identifier for the full audio file

    Returns:
        dict: Mapping of segment IDs to raw audio data
    """

    logging.info(f"Splitting audio for {full_audio_id}")
    split_audio = {}
    temp_audio_file = "temp_audio_in_memory.wav"
    with open(temp_audio_file, "wb") as f:
        f.write(audio_data)

    output_folder = f"data/split_audio/{full_audio_id}"

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    vad = pipeline(temp_audio_file)
    original_audio_segment = AudioSegment.from_file(temp_audio_file)
    original_audio_ndarray, sampling_rate = torchaudio.load(temp_audio_file)
    original_audio_ndarray = original_audio_ndarray[0]

    counter = 1
    for vad_span in vad.get_timeline().support():
        vad_segment = original_audio_segment[
            sec_to_millis(vad_span.start) : sec_to_millis(vad_span.end)  # noqa: E203
        ]
        vad_span_length = vad_span.end - vad_span.start
        if lower_limit <= vad_span_length <= upper_limit:
            segment_key = f"{full_audio_id}_{counter:04}"  # noqa: E231
            split_audio[segment_key] = vad_segment.raw_data
            save_segment(
                segment=vad_segment,
                folder=output_folder,
                prefix=full_audio_id,
                id=counter,
                start_ms=sec_to_millis(vad_span.start),
                end_ms=sec_to_millis(vad_span.end),
            )
            counter += 1
        elif vad_span_length > upper_limit:
            non_mute_segment_splits = librosa.effects.split(
                original_audio_ndarray[
                    int(
                        sec_to_frame(vad_span.start, sampling_rate)
                    ) : int(  # noqa: E203
                        sec_to_frame(vad_span.end, sampling_rate)
                    )
                ],
                top_db=30,
            )
            for split_start, split_end in non_mute_segment_splits:
                segment_split = original_audio_segment[
                    sec_to_millis(
                        vad_span.start + frame_to_sec(split_start, sampling_rate)
                    ) : sec_to_millis(  # noqa: E203
                        vad_span.start + frame_to_sec(split_end, sampling_rate)
                    )
                ]
                segment_split_duration = (
                    vad_span.start + frame_to_sec(split_end, sampling_rate)
                ) - (vad_span.start + frame_to_sec(split_start, sampling_rate))
                if lower_limit <= segment_split_duration <= upper_limit:
                    segment_key = f"{full_audio_id}_{counter:04}"  # noqa: E231
                    split_audio[segment_key] = segment_split.raw_data
                    save_segment(
                        segment=segment_split,
                        folder=output_folder,
                        prefix=full_audio_id,
                        id=counter,
                        start_ms=sec_to_millis(
                            vad_span.start + frame_to_sec(split_start, sampling_rate)
                        ),
                        end_ms=sec_to_millis(
                            vad_span.start + frame_to_sec(split_end, sampling_rate)
                        ),
                    )
                    counter += 1
                elif segment_split_duration > upper_limit:
                    chop_length = segment_split_duration / 2
                    while chop_length > upper_limit:
                        chop_length = chop_length / 2
                    for j in range(int(segment_split_duration / chop_length)):
                        segment_split_chop = original_audio_segment[
                            sec_to_millis(
                                vad_span.start
                                + frame_to_sec(split_start, sampling_rate)
                                + chop_length * j
                            ) : sec_to_millis(  # noqa: E203
                                vad_span.start
                                + frame_to_sec(split_start, sampling_rate)
                                + chop_length * (j + 1)
                            )
                        ]
                        segment_key = f"{full_audio_id}_{counter:04}"  # noqa: E231
                        split_audio[segment_key] = segment_split_chop.raw_data
                        save_segment(
                            segment=segment_split_chop,
                            folder=output_folder,
                            prefix=full_audio_id,
                            id=counter,
                            start_ms=sec_to_millis(
                                vad_span.start
                                + frame_to_sec(split_start, sampling_rate)
                                + chop_length * j
                            ),
                            end_ms=sec_to_millis(
                                vad_span.start
                                + frame_to_sec(split_start, sampling_rate)
                                + chop_length * (j + 1)
                            ),
                        )
                        counter += 1

    os.remove(temp_audio_file)
    logging.info(
        f"Finished splitting audio for {full_audio_id}. Total segments: {len(split_audio)}"
    )
    return split_audio
