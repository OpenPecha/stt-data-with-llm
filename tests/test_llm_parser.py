from stt_data_with_llm.LLM_post_corrector import get_LLM_corrected_text


def test_get_LLM_corrected_text():
    inference_transcript_file_path = "tests/text_files/llm_inference_data.txt"
    reference_transcript_file_path = "tests/text_files/llm_reference_data.txt"
    with open(inference_transcript_file_path, encoding="utf-8") as inference_file:
        inference_transcript_lines = [
            line.strip() for line in inference_file if line.strip()
        ]
    with open(reference_transcript_file_path, encoding="utf-8") as reference_file:
        reference_transcript_lines = [
            line.strip() for line in reference_file if line.strip()
        ]
    for (inference_transcript_line, reference_transcript_line) in zip(
        inference_transcript_lines, reference_transcript_lines
    ):
        # Get corrected text
        corrected_text = get_LLM_corrected_text(
            inference_transcript_line, reference_transcript_line
        )

        assert corrected_text is not None
