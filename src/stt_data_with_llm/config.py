# Catalog Parser
MAX_BYTES = 1024 * 1024
BACKUP_COUNT = 5


# Audio Segmentation
AUDIO_SEG_UPPER_LIMIT = 8
AUDIO_SEG_LOWER_LIMIT = 2


HYPER_PARAMETERS = {
    # onset/offset activation thresholds
    "onset": 0.5,
    "offset": 0.5,
    # remove speech regions shorter than that many seconds.
    "min_duration_on": 2.0,
    # fill non-speech regions shorter than that many seconds.
    "min_duration_off": 0.0,
}

# Define the headers (as given)
AUDIO_HEADERS = {
    "accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7",  # noqa: E501
    "accept-encoding": "gzip, deflate, br, zstd",
    "accept-language": "en-US,en;q=0.9,en-IN;q=0.8",
    "cache-control": "max-age=0",
    "cookie": "AMCVS_518ABC7455E462B97F000101%40AdobeOrg=1; s_cc=true; utag_main=v_id:019169eade56002296e6ea4a443c0507d001b075008f7$_sn:11$_se:5$_ss:0$_st:1727788387180$vapi_domain:rfa.org$ses_id:1727793787%3Bexp-session$_pn:5%3Bexp-session; AMCV_518ABC7455E462B97F000101%40AdobeOrg=1176715910%7CMCIDTS%7C19997%7CMCMID%7C92058839809215745258174654077801968713%7CMCAID%7CNONE%7CMCOPTOUT-1727793787s%7CNONE%7CvVersion%7C5.4.0; s_sq=%5B%5BB%5D%5D",  # noqa: E501
    "priority": "u=0, i",
    "sec-ch-ua": '"Microsoft Edge";v="129", "Not=A?Brand";v="8", "Chromium";v="129"',
    "sec-ch-ua-mobile": "?0",
    "sec-ch-ua-platform": '"Windows"',
    "sec-fetch-dest": "document",
    "sec-fetch-mode": "navigate",
    "sec-fetch-site": "none",
    "sec-fetch-user": "?1",
    "upgrade-insecure-requests": "1",
    "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/129.0.0.0 Safari/537.36 Edg/129.0.0.0",  # noqa: E501
}

# Inferfence
SAMPLE_RATE = 16000
CHANNELS = 1
SAMPLE_WIDTH = 2

API_URL = "https://wpgzw4at8o6876h0.us-east-1.aws.endpoints.huggingface.cloud"

# Validation
CER_THRESHOLD = 0.4
