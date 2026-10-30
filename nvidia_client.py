
import os
import requests
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("NVIDIA_API_KEY")
API_URL = "https://integrate.api.nvidia.com/v1/embeddings"
MODEL = "nvidia/nemotron-3-embed-1b"


def get_embedding(text, input_type):
    if not API_KEY:
        raise ValueError(
            "NVIDIA_API_KEY is missing. Add it to your .env file."
        )

    if input_type not in ("passage", "query"):
        raise ValueError("input_type must be passage or query")

    payload = {
        "input": [text],
        "model": MODEL,
        "input_type": input_type,
        "encoding_format": "float",
        "truncate": "NONE"
    }

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
        "Accept": "application/json"
    }

    response = requests.post(
        API_URL,
        headers=headers,
        json=payload,
        timeout=120
    )

    if not response.ok:
        raise RuntimeError(
            f"NVIDIA API error {response.status_code}: "
            f"{response.text[:1000]}"
        )

    result = response.json()
    return result["data"][0]["embedding"]
