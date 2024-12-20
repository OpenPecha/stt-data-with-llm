from util import get_audio, get_split_audio


def get_inference(audio_split_data):

    return inference_transcript


def get_inference_transcript(audio_url, full_audio_id):
    segment_upper_limit = 8
    segment_lower_limit = 2
    # Get audio from URL
    audio_data = get_audio(audio_url)  # Store audio in memory
    # Get split audio data
    split_audio_data = get_split_audio(
        audio_data, segment_upper_limit, segment_lower_limit
    )
    inference_transcript = []
    # Get inference transcript from the split audio data
    for split_data in split_audio_data:
        # upload to the s3 bucket
        # audio_split_url = upload_to_s3(split_data, full_audio_id)
        inference_transcript.append(get_inference(split_data))

    return inference_transcript
