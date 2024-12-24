import logging

import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO)


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
        print(df.head())
        # Log basic information about the DataFrame
        logging.info("Spreadsheet successfully read.")
        logging.info(f"Headers: {df.columns.tolist()}")
        logging.info(f"First few rows:\n{df.head()}")  # noqa

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
                logging.warning(f"Row missing 'ID': {row}")
                continue

            audio_transcription_datas[full_audio_id] = {
                "full_audio_id": full_audio_id,
                "sr_no": row.get("Sr. no", ""),
                "audio_url": row.get("Audio LInk", ""),
                "reference_transcript": row.get("Audio text link", ""),
                "speaker_id": row.get("Speaker ID", ""),
            }
        except Exception as e:
            logging.error(f"Error processing row: {row}. Error: {e}")

    logging.info(f"Parsed {len(audio_transcription_datas)} entries from the catalog.")
    return audio_transcription_datas
