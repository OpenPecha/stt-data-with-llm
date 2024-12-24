import json
import os

from src.stt_data_with_llm.catalog_parser import catalog_parser


def test_catalog_parser():
    # Parse the catalog

    """
    Main function to parse the catalog and save the audio transcription data as JSON.
    """
    # Replace with your actual spreadsheet ID
    google_spread_sheet_id = "1TXTmFTSbCuEy6nzZj--jjprFmZkpJZVz8gkMYrBcPUU"

    # Parse the catalog

    audio_transcription_datas = catalog_parser(google_spread_sheet_id)
    actual_output_dir = "tests/data/actual_output"
    if not os.path.exists(actual_output_dir):
        os.makedirs(actual_output_dir)
    # Save the audio transcription data as JSON
    actual_output_json_path = (
        actual_output_dir + "/actual_audio_transcription_data.json"
    )
    expected_output_json_path = (
        "tests/data/expected_output/audio_transcription_data.json"
    )
    with open(actual_output_json_path, "w", encoding="utf-8") as jsonfile:
        json.dump(audio_transcription_datas, jsonfile, ensure_ascii=False, indent=4)

    # Compare the actual and expected JSON files
    with open(actual_output_json_path, encoding="utf-8") as jsonfile:
        actual_output = json.load(jsonfile)
    with open(expected_output_json_path, encoding="utf-8") as jsonfile:
        expected_output = json.load(jsonfile)
    # compare the length actual and expected JSON files
    assert len(actual_output) == len(expected_output)


if __name__ == "__main__":
    test_catalog_parser()
