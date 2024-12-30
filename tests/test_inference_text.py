from stt_data_with_llm.audio_parser import get_audio, get_split_audio
from stt_data_with_llm.config import AUDIO_SEG_LOWER_LIMIT, AUDIO_SEG_UPPER_LIMIT
from stt_data_with_llm.inference_transcript import get_audio_inference_text


def test_inference_text():
    """
    Test function for the get_audio_inference_text functionality.
    """
    csv_file = "inference.csv"
    audio_urls = {
        "NW_001": "https://www.rfa.org/tibetan/sargyur/golok-china-religious-restriction-08202024054225.html/@@stream",  # noqa
        "NW_002": "https://www.rfa.org/tibetan/sargyur/vpn-china-restriction-08152024081404.html/@@stream",
    }
    inference_text = ""
    for seg_id, audio_url in audio_urls.items():
        audio_data = get_audio(audio_url)
        split_audio_data = get_split_audio(
            audio_data, seg_id, AUDIO_SEG_LOWER_LIMIT, AUDIO_SEG_UPPER_LIMIT
        )
        for audio_seg_id, audio_seg_data in split_audio_data.items():
            audio_seg_inference_transcript = get_audio_inference_text(audio_seg_data)
            inference_text += f"{audio_seg_inference_transcript}\n"

            with open(csv_file, "a") as file:
                file.write(inference_text)


if __name__ == "__main__":
    test_inference_text()
