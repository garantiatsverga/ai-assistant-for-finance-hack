# config.py
import os
from enum import Enum

class APIType(Enum):
    OPENROUTER = "openrouter"
    HACKATHON = "hackathon"

API_CONFIGS = {
    APIType.OPENROUTER: {
        "base_url": "https://openrouter.ai/api/v1",
        "api_key": os.getenv("EMBEDDER_API_KEY"),  # тот же ключ для всего
        "llm_model": "mistralai/mistral-7b-instruct",  # ТВОЯ РАБОЧАЯ МОДЕЛЬ
        "embedding_model": "text-embedding-3-small",
        "headers": {
            "HTTP-Referer": "https://finance-hackathon-test.com",
            "X-Title": "Finance Assistant"
        }
    },
    APIType.HACKATHON: {
        "base_url": "https://ai-for-finance-hack.up.railway.app/",
        "api_key": os.getenv("LLM_API_KEY"),
        "embedding_key": os.getenv("EMBEDDER_API_KEY"),
        "llm_model": "openrouter/mistralai/mistral-small-3.2-24b-instruct",
        "embedding_model": "text-embedding-3-small"
    }
}

def get_config(api_type: APIType):
    return API_CONFIGS[api_type]