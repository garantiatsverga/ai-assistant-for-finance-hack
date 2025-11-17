import numpy as np
from typing import List, Optional, Dict, Any
import logging
from openai import OpenAI
import time

logger = logging.getLogger(__name__)

class EmbeddingManager:
    """Менеджер для работы с эмбеддингами через API"""
    
    def __init__(self, api_key: str, base_url: str, model: str = "text-embedding-3-small", headers: dict = None, delay_per_request: float = 0.5):
        self.client = OpenAI(
            base_url=base_url,
            api_key=api_key
        )
        self.model = model
        self.embedding_dim = 1536
        self.headers = headers
        self.delay_per_request = delay_per_request

    def encode(self, text: str) -> Optional[np.ndarray]:
        """Получение эмбеддинга для одного текста"""
        if self.delay_per_request > 0:
            time.sleep(self.delay_per_request)
        try:
            response = self.client.embeddings.create(
                model=self.model,
                input=[text[:8000]]
            )
            return np.array(response.data[0].embedding)
        except Exception as e:
            logger.error(f"Ошибка получения эмбеддинга: {e}")
            return None

    def encode_batch(self, texts: List[str]) -> Optional[List[np.ndarray]]:
        """Получение эмбеддингов для батча текстов (настоящий батч)"""
        if not texts:
            return []
        if self.delay_per_request > 0:
            time.sleep(self.delay_per_request)
        try:
            truncated_texts = [text[:8000] for text in texts]
            response = self.client.embeddings.create(
                model=self.model,
                input=truncated_texts
            )
            embeddings = [np.array(item.embedding) for item in response.data]
            return embeddings
        except Exception as e:
            logger.error(f"Ошибка получения эмбеддингов батчом: {e}")
            return None

    def get_status(self) -> Dict[str, Any]:
        """Статус менеджера"""
        return {
            'model': self.model,
            'embedding_dim': self.embedding_dim,
            'status': 'active'
        }