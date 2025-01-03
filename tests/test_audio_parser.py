import json
import logging
from unittest import TestCase, mock

from stt_data_with_llm.audio_parser import get_audio, get_split_audio
from stt_data_with_llm.config import AUDIO_SEG_LOWER_LIMIT, AUDIO_SEG_UPPER_LIMIT


class TestGetSplitAudio(TestCase):
    @mock.patch("stt_data_with_llm.audio_parser.initialize_vad_pipeline")
    @mock.patch("stt_data_with_llm.audio_parser.Pipeline")
    def test_get_split_audio(self, mock_pipeline, mock_initialize_vad):
        """
        Test function for the get_split_audio functionality.
        """
        # Define mock VAD outputs for each audio file
        vad_outputs = {
            "NW_001": "./tests/data/vad_output/NW_001_vad_output.json",
            "NW_002": "./tests/data/vad_output/NW_002_vad_output.json",
            "NW_003": "./tests/data/vad_output/NW_003_vad_output.json",
        }
        # Load all VAD outputs dynamically
        mock_vad_results = {}
        for seg_id, vad_path in vad_outputs.items():
            with open(vad_path, encoding="utf-8") as file:
                mock_vad_results[seg_id] = json.load(file)

        class MockVADPipeline:
            def __init__(self, seg_id):
                self.seg_id = seg_id

            def __call__(self, audio_file):
                return MockVADResult(self.seg_id)

        class MockVADResult:
            def __init__(self, seg_id):
                self.vad_output = mock_vad_results[seg_id]

            def get_timeline(self):
                class MockTimeline:
                    def __init__(self, timeline):
                        self.timeline = timeline

                    def support(self):
                        return [
                            type(
                                "Segment",
                                (),
                                {"start": seg["start"], "end": seg["end"]},
                            )
                            for seg in self.timeline
                        ]

                return MockTimeline(self.vad_output["timeline"])

        # Setup mock behavior
        def mock_initialize_pipeline(seg_id):
            try:
                return MockVADPipeline(seg_id)
            except Exception as e:
                logging.warning(
                    f"Mocking failed: {e}. Falling back to actual function."
                )
                return None

        audio_urls = {
            "NW_001": "https://www.rfa.org/tibetan/sargyur/golok-china-religious-restriction-08202024054225.html/@@stream",  # noqa
            "NW_002": "https://vot.org/wp-content/uploads/2024/03/tc88888888888888.mp3",
            "NW_003": "https://voa-audio-ns.akamaized.net/vti/2024/04/13/01000000-0aff-0242-a7bb-08dc5bc45613.mp3",
        }
        num_of_seg_in_audios = {}
        for seg_id, audio_url in audio_urls.items():
            mock_pipeline = mock_initialize_pipeline(seg_id)
            if mock_pipeline:
                mock_initialize_vad.return_value = mock_pipeline
            else:
                mock_initialize_vad.side_effect = None  # Disable the mock for fallback

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
    TestGetSplitAudio().test_get_split_audio()
