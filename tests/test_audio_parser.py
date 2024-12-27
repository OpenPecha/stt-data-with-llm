import json
import os

from stt_data_with_llm.audio_parser import get_audio, get_split_audio
from stt_data_with_llm.catalog_parser import catalog_parser
from stt_data_with_llm.config import AUDIO_SEG_LOWER_LIMIT, AUDIO_SEG_UPPER_LIMIT


def test_get_split_audio():
    # Replace with your actual spreadsheet ID
    google_spread_sheet_id = "1nE2AVpOr6dIF8iZlPhOfJRMCSI25fYHswxdNIAYuZ7M"

    actual_json_dir = "tests/data/actual_json"
    if not os.path.exists(actual_json_dir):
        os.makedirs(actual_json_dir)
    actual_json_file = os.path.join(actual_json_dir, "actual_json_audio_data.json")
    audio_seg = {}
    # Parse the catalog
    number_of_full_audio_for_testing = 2
    number_of_full_audio_counter = 0
    audio_transcription_datas = catalog_parser(google_spread_sheet_id)
    for data_id, audio_data_info in audio_transcription_datas.items():
        if number_of_full_audio_counter < number_of_full_audio_for_testing:
            audio_url = audio_data_info.get("audio_url", "")
            full_audio_id = audio_data_info.get("full_audio_id", "")
            audio_data = get_audio(audio_url)
            split_audio_data = get_split_audio(
                audio_data, full_audio_id, AUDIO_SEG_LOWER_LIMIT, AUDIO_SEG_UPPER_LIMIT
            )
            split_count = 0
            split_count = len(split_audio_data)
            audio_seg[full_audio_id] = split_count
        number_of_full_audio_counter += 1

    with open(actual_json_file, "a") as file:
        json.dump(audio_seg, file, indent=4)
    expected_json_file = "tests/data/expected_json/expected_json_audio_data.json"
    with open(expected_json_file) as file:
        expected_json = json.load(file)
    with open(actual_json_file) as file:
        actual_json = json.load(file)
    for num_seg in expected_json:
        assert actual_json[num_seg] == expected_json[num_seg]


if __name__ == "__main__":
    test_get_split_audio()
