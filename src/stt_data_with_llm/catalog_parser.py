import json
import logging

import requests


def parse_catalog(catalog_url):
    """Reads a catalog from a URL and parses the catalog.

    Args:
        catalog_url (str): The URL of the catalog.

    Returns:
        dict: A dictionary of dictionaries containing the parsed catalog data.
    """
    try:
        response = requests.get(catalog_url)
        catalog_data = response.json()
        parsed_items = {}
        for catalog_item in catalog_data:
            parsed_item = {
                "sr_no": catalog_item.get("Sr.no"),
                "id": catalog_item.get("ID"),
                "audio_url": catalog_item.get("Audio URL"),
                "audio_text": catalog_item.get("Audio Text"),
                "speaker_name": catalog_item.get("Speaker Name"),
                "news_channel": catalog_item.get("News Channel"),
                "publishing_year": catalog_item.get("Publishing Year"),
            }
            parsed_items[catalog_item.get("ID")] = parsed_item

        return parsed_items
    except requests.RequestException as e:
        logging.error(f"Error fetching Catalog: {e}")
        return {}
    except json.JSONDecodeError as e:
        logging.error(f"Error parsing Catalog JSON: {e}")
        return {}
