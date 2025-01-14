from stt_data_with_llm.audio_parser import get_audio, get_split_audio
from stt_data_with_llm.config import AUDIO_SEG_LOWER_LIMIT, AUDIO_SEG_UPPER_LIMIT
from stt_data_with_llm.main import upload_to_s3


def test_upload_to_s3():
    audio_urls = {
        "NW_001": "https://www.rfa.org/tibetan/sargyur/golok-china-religious-restriction-08202024054225.html/@@stream",  # noqa
    }
    s3_bucket_name = "monlam.ai.stt"
    for seg_id, audio_url in audio_urls.items():

        audio_data = get_audio(audio_url)
        split_audio_data = get_split_audio(
            audio_data, seg_id, AUDIO_SEG_LOWER_LIMIT, AUDIO_SEG_UPPER_LIMIT
        )
        for (split_seg_id, audio_seg_data) in split_audio_data.items():
            file_name = f"testing/{split_seg_id}.wav"
            upload_to_s3(s3_bucket_name, file_name, audio_seg_data)


if __name__ == "__main__":
    test_upload_to_s3()
