import json

from stt_data_with_llm.audio_parser import get_audio, get_split_audio
from stt_data_with_llm.config import AUDIO_SEG_LOWER_LIMIT, AUDIO_SEG_UPPER_LIMIT


def test_get_split_audio():
    """
    Test function for the get_split_audio functionality.
    """
    audio_urls = {
        "NW_001": "https://www.rfa.org/tibetan/sargyur/golok-china-religious-restriction-08202024054225.html/@@stream",
        "NW_002": "https://www.rfa.org/tibetan/sargyur/golok-china-religious-restriction-08202024054225.html/@@stream",  # noqa
        "NW_003": "https://www.rfa.org/tibetan/sargyur/vpn-china-restriction-08152024081404.html/@@stream",
    }
    seg_audio = {}
    for seg_id, audio_url in audio_urls.items():

        audio_data = get_audio(audio_url)
        split_audio_data = get_split_audio(
            audio_data, seg_id, AUDIO_SEG_LOWER_LIMIT, AUDIO_SEG_UPPER_LIMIT
        )
        num_split = len(split_audio_data)
        seg_audio[seg_id] = num_split
    expected_num_split_file = "tests/data/expected_audio_data.json"
    with open(expected_num_split_file, encoding="utf-8") as file:
        expected_num_split = json.load(file)
    assert seg_audio == expected_num_split


if __name__ == "__main__":
    test_get_split_audio()
