import logging

import pandas as pd

# Call the setup_logging function at the beginning of your script


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
        df = pd.read_csv(url, header=0, encoding="utf-8")
        # Log basic information about the DataFrame
        logging.info("Spreadsheet successfully read.")
        logging.info(f"Headers: {df.columns.tolist()}")

        return df
    except Exception as e:
        logging.error(f"Error reading spreadsheet: {e}")
        return pd.DataFrame()


def parse_catalog(google_sheet_id, start_sr_no=None, end_sr_no=None):
    """
    Parses an audio transcription catalog from a Google Spreadsheet within a specified range of Sr.no values.

    Args:
        google_sheet_id (str): The ID of the Google Spreadsheet containing the audio transcription catalog.
        start_sr_no (int, optional): The starting Sr.no to process. If None, starts from the beginning.
        end_sr_no (int, optional): The ending Sr.no to process. If None, processes until the end.

    Returns:
        dict: A dictionary where keys are unique IDs (e.g., "full_audio_id") and values are dictionaries of audio data.
    """

    catalog_df = read_spreadsheet(google_sheet_id)

    # Check if the catalog DataFrame is empty
    if catalog_df.empty:
        logging.warning("Catalog DataFrame is empty.")
        return {}
    # Convert Sr.no column to numeric, replacing any non-numeric values with NaN
    catalog_df["Sr.no"] = pd.to_numeric(catalog_df["Sr.no"], errors="coerce")

    # Filter the DataFrame based on the specified range
    if start_sr_no is not None:
        catalog_df = catalog_df[catalog_df["Sr.no"] >= start_sr_no]
    if end_sr_no is not None:
        catalog_df = catalog_df[catalog_df["Sr.no"] <= end_sr_no]

    audio_transcription_catalog = {}

    for index, row in catalog_df.iterrows():
        try:
            full_audio_id = row.get("ID", "")
            if not full_audio_id:
                logging.warning(f"Row missing 'ID': {row.to_dict()}")

            audio_transcription_catalog[str(index)] = {
                "full_audio_id": full_audio_id if not pd.isna(full_audio_id) else "",
                "sr_no": row.get("Sr.no", "")
                if not pd.isna(row.get("Sr.no", ""))
                else "",
                "audio_url": row.get("Audio URL", "")
                if not pd.isna(row.get("Audio URL", ""))
                else "",
                "reference_transcript": row.get("Audio Text", "")
                if not pd.isna(row.get("Audio Text", ""))
                else "",
                "speaker_name": row.get("Speaker Name", "")
                if not pd.isna(row.get("Speaker Name", ""))
                else "",
                "speaker_gender": row.get("Speaker Gender", "")
                if not pd.isna(row.get("Speaker Gender", ""))
                else "",
                "news_channel": row.get("News Channel", "")
                if not pd.isna(row.get("News Channel", ""))
                else "",
                "publishing_year": row.get("Publishing Year", "")
                if not pd.isna(row.get("Publishing Year", ""))
                else "",
            }

        except Exception as e:
            logging.error(f"Error processing row: {row.to_dict()}. Error: {e}")

    logging.info(f"Parsed {len(audio_transcription_catalog)} entries from the catalog.")
    return audio_transcription_catalog
