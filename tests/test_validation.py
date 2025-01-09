import logging

from stt_data_with_llm.main import (
    get_inference_transcript,
    get_original_text,
    is_valid_transcript,
)


def test_validation():
    original_text_file_path = "tests/data/original_text_file.txt"
    inference_transcript_file_path = "tests/data/inference_transcript.txt"
    with open(original_text_file_path, encoding="utf-8") as file:
        original_text = file.read()
    with open(inference_transcript_file_path, encoding="utf-8") as file:
        inference_transcript = file.read()
    validation_original_text = get_original_text(original_text)
    validation_inference_transcript = get_inference_transcript(inference_transcript)
    logging.info(validation_original_text)
    logging.info(validation_inference_transcript)
    assert is_valid_transcript(
        validation_inference_transcript, validation_original_text
    )


if __name__ == "__main__":
    test_validation()
