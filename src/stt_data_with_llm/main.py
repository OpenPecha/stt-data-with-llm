import logging

from fast_antx.core import transfer

from stt_data_with_llm.catalog_parser import parse_catalog
from stt_data_with_llm.config import AUDIO_SEG_UPPER_LIMIT, AUDIO_SEG_LOWER_LIMIT
from stt_data_with_llm.audio_parser import get_audio, get_split_audio
from stt_data_with_llm.LLM_post_corrector import get_LLM_corrected_text
from stt_data_with_llm.inference_transcript import get_audio_inference_text


logging.basicConfig(filename='./pipeline.log', level=logging.INFO)

def transfer_segmentation(inference_transcript, reference_transcript):
    reference_transcript = reference_transcript.replace("\n", " ")
    patterns = [['segmentation', '(\n)']]
    reference_transcript_with_inference_segmentation = transfer(inference_transcript, patterns, reference_transcript)
    return reference_transcript_with_inference_segmentation

def is_valid_transcript(inference_transcript, reference_transcript):
    pass

def post_process_audio_transcript_pairs(audio_data_info):
    post_processed_audio_transcript_pairs = {}
    inference_transcript = ""
    audio_url = audio_data_info.get("audio_url", "")
    full_audio_id = audio_data_info.get("full_audio_id", "")
    reference_transcript = audio_data_info.get("reference_transcript", "")
    if not audio_url:
        return None,full_audio_id
    audio_data = get_audio(audio_url)
    split_audio_data = get_split_audio(audio_data, AUDIO_SEG_LOWER_LIMIT, AUDIO_SEG_UPPER_LIMIT, full_audio_id)
    for audio_seg_id, audio_seg_data in split_audio_data.items():
        audio_seg_inference_transcript = get_audio_inference_text(audio_seg_data)
        inference_transcript += f'{audio_seg_inference_transcript}\n'
    if not is_valid_transcript(inference_transcript, reference_transcript):
        return None,full_audio_id
    reference_transcript_with_inference_segmentation = transfer_segmentation(inference_transcript, reference_transcript)
    inference_transcripts = inference_transcript.split("\n")
    reference_transcripts = reference_transcript_with_inference_segmentation.split("\n")
    for seg_walker,(audio_seg_id, audio_seg_data) in enumerate(split_audio_data.items()):
        seg_inference_text = inference_transcripts[seg_walker]
        seg_reference_text = reference_transcripts[seg_walker]
        seg_LLM_corrected_text = get_LLM_corrected_text(seg_inference_text, seg_reference_text)
        post_process_audio_transcript_pairs[audio_seg_id] = {
            "audio_seg_data": audio_seg_data,
            "inference_transcript": seg_inference_text,
            "reference_transcript": seg_reference_text,
            "LLM_corrected_text": seg_LLM_corrected_text
        }
    return post_processed_audio_transcript_pairs,full_audio_id


def save_post_processed_audio_transcript_pairs(post_processed_audio_transcript_pairs, audio_data_info):
    # Save post processed audio transcript pairs in csv
    pass


def get_audio_transcript_pairs(audio_transcription_catalog_url):
    audio_transcription_datas = parse_catalog(audio_transcription_catalog_url)
    for data_id, audio_data_info in audio_transcription_datas.items():
        post_processed_audio_transcript_pairs,full_audio_id = post_process_audio_transcript_pairs(audio_data_info)
        if post_processed_audio_transcript_pairs:
            save_post_processed_audio_transcript_pairs(post_processed_audio_transcript_pairs, audio_data_info)
        else:
            logging.info(f"Audio data with ID {full_audio_id} has invalid transcript")


