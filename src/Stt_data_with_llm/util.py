def get_audio(audio_url):
    # Get audio from URL
    return audio_data


def get_split_audio(audio_url, segment_upper_limit, segment_lower_limit):
    audio_data = get_audio(audio_url)  # Store audio in memory


def upload_to_s3(split_audio_data, full_audio_id):
    # upload to the s3 bucket
    return audio_split_url


def save_inference_transcript_to_csv(
    full_audio_id, audio_split_url, inference_transcript
):
    # Save the inference transcript to their corresponding
    print("Saving inference transcript")
