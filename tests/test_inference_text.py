import json
from unittest import TestCase, mock

from stt_data_with_llm.audio_parser import get_audio, get_split_audio
from stt_data_with_llm.config import AUDIO_SEG_LOWER_LIMIT, AUDIO_SEG_UPPER_LIMIT
from stt_data_with_llm.inference_transcript import get_audio_inference_text


class TestInferenceText(TestCase):
    @mock.patch("stt_data_with_llm.inference_transcript.query_audio_api")
    def test_inference_text(self, mock_query_audio_api):
        """
        Test function for the get_audio_inference_text functionality.
        """
        mock_api_responses = []
        # Load the mock query_audio_api response from the saved json file
        with open(
            "tests/query_audio_api_output/api_response.json", encoding="utf-8"
        ) as file:
            for line in file:
                mock_api_responses.append(json.loads(line.strip()))
        # Set the return value for the mock query_audio_api
        mock_query_audio_api.side_effect = mock_api_responses
        audio_urls = {
            "NW_001": "https://www.rfa.org/tibetan/sargyur/golok-china-religious-restriction-08202024054225.html/@@stream",  # noqa
            "NW_002": "https://voa-audio-ns.akamaized.net/vti/2024/08/02/01000000-c0a8-0242-b72e-08dcb2d4e87a.mp3",
            "NW_003": "https://vot.org/wp-content/uploads/2024/02/tl-21022024.mp3",
        }
        inference_text = ""
        audio_seg_counter = 0
        inference_text_counter = 0
        for seg_id, audio_url in audio_urls.items():
            audio_data = get_audio(audio_url)
            split_audio_data = get_split_audio(
                audio_data, seg_id, AUDIO_SEG_LOWER_LIMIT, AUDIO_SEG_UPPER_LIMIT
            )
            audio_seg_counter += len(split_audio_data)
            for audio_seg_id, audio_seg_data in split_audio_data.items():
                audio_seg_inference_transcript = get_audio_inference_text(
                    audio_seg_data
                )
                inference_text += f"{audio_seg_inference_transcript}\n"
                inference_text_counter += 1

        assert audio_seg_counter == inference_text_counter


if __name__ == "__main__":
    TestInferenceText.test_inference_text()
