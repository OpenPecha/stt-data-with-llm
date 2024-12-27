import json

from stt_data_with_llm.catalog_parser import catalog_parser


def test_catalog_parser():
    # Parse the catalog

    """
    Main function to parse the catalog and save the audio transcription data as JSON.
    """
    # Replace with your actual spreadsheet ID
    google_spread_sheet_id = "14pCi8pxD_Ms3i3RAcBWNrCT9MocnRKD49jTLxDHzDe0"

    # Parse the catalog

    audio_transcription_catalog = catalog_parser(google_spread_sheet_id)
    expected_output_json_path = "tests/data/expected_catalog_data.json"
    with open(expected_output_json_path, encoding="utf-8") as file:
        expected_output_json = json.load(file)

    assert audio_transcription_catalog == expected_output_json


if __name__ == "__main__":
    test_catalog_parser()
