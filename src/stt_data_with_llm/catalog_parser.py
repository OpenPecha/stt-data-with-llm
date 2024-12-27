import logging

import pandas as pd

from stt_data_with_llm.util import setup_logging

# Call the setup_logging function at the beginning of your script
setup_logging("catalog_parse.log")


def read_spreadsheet(sheet_id):
    """
    Reads a Google Spreadsheet as a Pandas DataFrame without mixing rows and headers.

    Args:
        sheet_id (str): The ID of the Google Spreadsheet.

    Returns:
        pd.DataFrame: A cleaned DataFrame with rows and headers properly separated.
    """
    url = (
        f"https://docs.google.com/spreadsheets/d/{sheet_id}/gviz/tq?tqx=out:csv"  # noqa
    )
    try:
        # Read the CSV data from the Google Spreadsheet
        df = pd.read_csv(url, header=0)
        # Log basic information about the DataFrame
        logging.info("Spreadsheet successfully read.")
        logging.info(f"Headers: {df.columns.tolist()}")

        return df
    except Exception as e:
        logging.error(f"Error reading spreadsheet: {e}")
        return pd.DataFrame()


def catalog_parser(audio_url):
    """
    Parses an audio transcription catalog from a Google Spreadsheet.

    Args:
        audio_url (str): The URL of the Google Spreadsheet containing the audio transcription catalog.

    Returns:
        dict: A dictionary where keys are unique IDs (e.g., "full_audio_id") and values are dictionaries of audio data.
    """
    catalog_df = read_spreadsheet(audio_url)

    # Check if the catalog DataFrame is empty
    if catalog_df.empty:
        logging.warning("Catalog DataFrame is empty.")
        return {}

    audio_transcription_datas = {}
    for _, row in catalog_df.iterrows():
        try:
            full_audio_id = row.get("ID", "")
            if not full_audio_id:
                logging.warning(f"Row missing 'ID': {row.to_dict()}")
                continue

            audio_transcription_datas[full_audio_id] = {
                "full_audio_id": full_audio_id,
                "sr_no": row.get("Sr.no", ""),
                "audio_url": row.get("Audio URL", ""),
                "reference_transcript": row.get("Audio Text", ""),
                "speaker_name": row.get("Speaker Name", ""),
                "speaker_gender": row.get("Speaker Gender", ""),
                "news_channel": row.get("News Channel", ""),
                "publishing_year": row.get("Publishing Year", ""),
            }
            logging.info(f"print full audio id: %s" % full_audio_id)  # noqa: F541
        except Exception as e:
            logging.error(f"Error processing row: {row.to_dict()}. Error: {e}")

    logging.info(f"Parsed {len(audio_transcription_datas)} entries from the catalog.")
    return audio_transcription_datas
