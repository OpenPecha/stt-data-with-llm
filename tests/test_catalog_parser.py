import json

from stt_data_with_llm.catalog_parser import parse_catalog


def test_catalog_parser():
    # Parse the catalog

    """
    Main function to parse the catalog and save the audio transcription data as JSON.
    """
    # Replace with your actual spreadsheet ID
    google_spread_sheet_id = "1Iy01o2hsrhWpbOQzFfC1gOVqw4j1AMp7poEU2eu7WN0"

    # Parse the catalog

    audio_transcription_catalog = parse_catalog(google_spread_sheet_id, 1, 20)
    expected_output_json_path = "tests/data/expected_catalog_data.json"
    with open(expected_output_json_path, encoding="utf-8") as file:
        expected_output = json.load(file)
    assert audio_transcription_catalog == expected_output


if __name__ == "__main__":
    test_catalog_parser()
