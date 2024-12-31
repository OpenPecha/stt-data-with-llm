import json

from stt_data_with_llm.audio_parser import get_audio, get_split_audio
from stt_data_with_llm.config import AUDIO_SEG_LOWER_LIMIT, AUDIO_SEG_UPPER_LIMIT


def test_get_split_audio():
    """
    Test function for the get_split_audio functionality.
    """
    audio_urls = {
        "NW_001": "https://www.rfa.org/tibetan/sargyur/golok-china-religious-restriction-08202024054225.html/@@stream",  # noqa
        "NW_002": "https://vot.org/wp-content/uploads/2024/03/tc88888888888888.mp3",
        "NW_003": "https://voa-audio-ns.akamaized.net/vti/2024/04/13/01000000-0aff-0242-a7bb-08dc5bc45613.mp3",
    }
    num_of_seg_in_audios = {}
    for seg_id, audio_url in audio_urls.items():

        audio_data = get_audio(audio_url)
        split_audio_data = get_split_audio(
            audio_data, seg_id, AUDIO_SEG_LOWER_LIMIT, AUDIO_SEG_UPPER_LIMIT
        )
        num_split = len(split_audio_data)
        num_of_seg_in_audios[seg_id] = num_split
    expected_num_of_seg_in_audios = "tests/data/expected_audio_data.json"
    with open(expected_num_of_seg_in_audios, encoding="utf-8") as file:
        expected_num_split = json.load(file)
    assert num_of_seg_in_audios == expected_num_split


if __name__ == "__main__":
    test_get_split_audio()
