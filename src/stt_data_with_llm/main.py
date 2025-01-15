import csv
import io
import logging

import boto3
from fast_antx.core import transfer

from stt_data_with_llm.audio_parser import get_audio, get_split_audio
from stt_data_with_llm.catalog_parser import parse_catalog
from stt_data_with_llm.config import (
    AUDIO_SEG_LOWER_LIMIT,
    AUDIO_SEG_UPPER_LIMIT,
    CER_THRESHOLD,
)
from stt_data_with_llm.inference_transcript import get_audio_inference_text
from stt_data_with_llm.LLM_post_corrector import get_LLM_corrected_text
from stt_data_with_llm.util import (
    calculate_cer,
    get_inference_transcript,
    get_original_text,
)

logging.basicConfig(filename="./pipeline.log", level=logging.INFO)


def transfer_segmentation(inference_transcript, reference_transcript):
    """Transfers the segmentation patterns from the inference transcript to the reference transcript.

    Args:
        inference_transcript (str): The transcript generated by the inference model.
        reference_transcript (str): original reference text

    Returns:
        str: The reference transcript with segmentation from the inference transcript applied.
    """
    reference_transcript = reference_transcript.replace("\n", " ")
    patterns = [["segmentation", "(\n)"]]
    reference_transcript_with_inference_segmentation = transfer(
        inference_transcript, patterns, reference_transcript
    )
    return reference_transcript_with_inference_segmentation


def audio_segment_to_bytes(audio_segment):
    buffer = io.BytesIO()
    audio_segment.export(buffer, format="wav")
    audio_data = buffer.getvalue()
    return audio_data


def is_valid_transcript(inference_transcript, reference_transcript):
    """Validates the reference transcript by comparing it to the inference transcript
    using the Character Error Rate (CER) metric.

    Args:
        inference_transcript (str): The transcript generated by the inference model.
        reference_transcript (str): Original reference transcript

    Returns:
        bool:
            -`True` if the cer value is less than the `CER_THRESHOLD`
            -`False` otherwise
    """

    cer_value = calculate_cer(inference_transcript, reference_transcript)
    logging.info(f"Cer Value: {cer_value}")
    return cer_value <= CER_THRESHOLD


def post_process_audio_transcript_pairs(audio_data_info):
    post_processed_audio_transcript_pairs = {}
    inference_transcript = ""
    audio_url = audio_data_info.get("audio_url", "")
    full_audio_id = audio_data_info.get("full_audio_id", "")
    reference_transcript = audio_data_info.get("reference_transcript", "")
    if not audio_url:
        return None, full_audio_id

    audio_data = get_audio(audio_url)
    split_audio_data = get_split_audio(
        audio_data, full_audio_id, AUDIO_SEG_LOWER_LIMIT, AUDIO_SEG_UPPER_LIMIT
    )
    for audio_seg_id, audio_seg_data in split_audio_data.items():
        audio_seg_inference_transcript = get_audio_inference_text(audio_seg_data)
        inference_transcript += f"{audio_seg_inference_transcript}\n"
    validation_original_text = get_original_text(reference_transcript)
    validation_inference_transcript = get_inference_transcript(inference_transcript)
    if not is_valid_transcript(
        validation_inference_transcript, validation_original_text
    ):
        return None, full_audio_id
    reference_transcript_with_inference_segmentation = transfer_segmentation(
        inference_transcript, reference_transcript
    )
    inference_transcripts = inference_transcript.split("\n")
    reference_transcripts = reference_transcript_with_inference_segmentation.split("\n")

    for seg_walker, (audio_seg_id, audio_seg_data) in enumerate(
        split_audio_data.items()
    ):
        seg_inference_text = inference_transcripts[seg_walker]
        seg_reference_text = reference_transcripts[seg_walker]
        if not is_valid_transcript(seg_inference_text, seg_reference_text):
            seg_LLM_corrected_text = get_LLM_corrected_text(seg_inference_text, False)
        else:
            seg_LLM_corrected_text = get_LLM_corrected_text(
                seg_inference_text, True, seg_reference_text
            )
        post_process_audio_transcript_pairs[audio_seg_id] = {
            "audio_seg_data": audio_seg_data,
            "inference_transcript": seg_inference_text,
            "reference_transcript": seg_reference_text,
            "LLM_corrected_text": seg_LLM_corrected_text,
        }
    return post_processed_audio_transcript_pairs, full_audio_id


def extract_duration_from_filename(file_name):
    """
    Extracts start_ms and end_ms from the file_name and calculates the duration in seconds.

    Args:
        file_name (str): The file name in the format "full_audio_id_counter_start_ms_to_end_ms".

    Returns:
        float: The duration of the audio segment in seconds.
    """
    try:
        # Extract start_ms and end_ms from the file_name
        parts = file_name.split("_")
        start_ms = int(parts[-3])
        end_ms = int(parts[-1])  # Last part is end_ms

        # Calculate duration
        duration_ms = end_ms - start_ms
        return round(duration_ms / 1000, 2)
    except Exception as e:
        logging.error(f"Error extracting duration from file name {file_name}: {e}")
        return 0.0  # Default to 0 if there's an error


def upload_to_s3(bucket_name, file_name, audio_segment):
    session = boto3.Session()
    s3 = session.client("s3")
    try:
        # Convert AudioSegment to bytes
        audio_data = audio_segment_to_bytes(audio_segment)

        # Upload the file to S3
        s3.put_object(
            Bucket=bucket_name,
            Key=file_name,
            Body=audio_data,
            ContentType="audio/wav",
            ContentDisposition="inline",
        )

        # Generate the CloudFront URL
        cleaned_file_name = (
            file_name.split("/", 1)[1] if "/" in file_name else file_name
        )
        cloudfront_url = f"https://d38pmlk0v88drf.cloudfront.net/stt_news_auto_data/{cleaned_file_name}"  # noqa
        logging.info(f"File uploaded to S3 and accessible at: {cloudfront_url}")
        return cloudfront_url
    except Exception as e:
        logging.error(f"Error uploading file to S3: {e}")
        return None


def save_post_processed_audio_transcript_pairs(
    post_processed_audio_transcript_pairs, audio_data_info
):
    output_file = "processed_audio_transcript.csv"
    # Define Csv column headers
    headers = [
        "file_name",
        "audio_url",
        "inference_transcript",
        "audio_duration",
        "speaker_name",
        "speaker_gender",
        "news_channel",
        "publishing_year",
    ]
    # Extract the metadata from the catalog
    speaker_name = audio_data_info.get("speaker_name", "")
    speaker_gender = audio_data_info.get("speaker_gender", "")
    news_channel = audio_data_info.get("news_channel", "")
    publishing_year = audio_data_info.get("publishing_year", "")
    # S3 bucket name
    s3_bucket_name = "monlam.ai.stt"

    try:
        with open(output_file, "w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=headers)
            writer.writeheader()
            for (
                audio_seg_id,
                audio_seg_data,
            ) in post_processed_audio_transcript_pairs.items():

                # Prepare file name and upload to S3
                file_name = f"stt_news_auto_data/{audio_seg_id}.wav"
                audio_url = upload_to_s3(
                    s3_bucket_name, file_name, audio_seg_data["audio_seg_data"]
                )

                # Extract duration from the audio_seg_id (file_name)
                audio_duration = extract_duration_from_filename(audio_seg_id)
                # Write a row to the CSV file
                writer.writerow(
                    {
                        "filename": audio_seg_id,
                        "audio_url": audio_url,
                        "inference_transcript": audio_seg_data["inference_transcript"],
                        "audio_duration": audio_duration,
                        "speaker_name": speaker_name,
                        "speaker_gender": speaker_gender,
                        "news_channel": news_channel,
                        "publishing_year": publishing_year,
                    }
                )
        logging.info(f"Processed audio transcript pairs saved to {output_file}")
    except Exception as e:
        logging.error(f"Error saving processed audio transcript pairs: {e}")


def get_audio_transcript_pairs(
    audio_transcription_catalog_url, start_sr_no=None, end_sr_no=None
):
    audio_transcription_datas = parse_catalog(
        audio_transcription_catalog_url, start_sr_no, end_sr_no
    )
    for data_id, audio_data_info in audio_transcription_datas.items():
        (
            post_processed_audio_transcript_pairs,
            full_audio_id,
        ) = post_process_audio_transcript_pairs(audio_data_info)
        if post_processed_audio_transcript_pairs:
            save_post_processed_audio_transcript_pairs(
                post_processed_audio_transcript_pairs, audio_data_info
            )
        else:
            logging.info(f"Audio data with ID {full_audio_id} has invalid transcript")


if __name__ == "__main__":
    # Replace with your actual spreadsheet ID
    google_spread_sheet_id = "1Iy01o2hsrhWpbOQzFfC1gOVqw4j1AMp7poEU2eu7WN0"
    get_audio_transcript_pairs(google_spread_sheet_id, 1, 20)
