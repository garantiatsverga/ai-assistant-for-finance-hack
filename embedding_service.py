# embedding_service.py
import numpy as np
from typing import List, Optional, Dict, Any
import logging
from openai import OpenAI
import time

logger = logging.getLogger(__name__)

class EmbeddingManager:
    """Менеджер для работы с эмбеддингами через API"""
    
    def __init__(self, api_key: str, base_url: str, model: str = "text-embedding-3-small", headers: dict = None, delay_per_request: float = 0.2):
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
        # Задержка перед запросом
        if self.delay_per_request > 0:
            time.sleep(self.delay_per_request)
        try:
            response = self.client.embeddings.create(
                model=self.model,
                input=text[:8000]  # Ограничение длины
            )
            return np.array(response.data[0].embedding)
        except Exception as e:
            logger.error(f"Ошибка получения эмбеддинга: {e}")
            return None
    
    def encode_batch(self, texts: List[str]) -> Optional[List[np.ndarray]]:
        """Получение эмбеддингов для батча текстов"""
        embeddings = []
        
        for i, text in enumerate(texts):
            # Задержка перед каждым запросом в батче
            if self.delay_per_request > 0:
                time.sleep(self.delay_per_request)
            try:
                embedding = self.encode(text) # вызов с задержкой
                if embedding is not None:
                    embeddings.append(embedding)
                else:
                    # Создаем нулевой эмбеддинг в случае ошибки
                    embeddings.append(np.zeros(self.embedding_dim))
                    logger.warning(f"Создан нулевой эмбеддинг для текста {i}")
                    
            except Exception as e:
                logger.error(f"Ошибка обработки текста {i}: {e}")
                embeddings.append(np.zeros(self.embedding_dim))
        
        return embeddings if embeddings else None
    
    def get_status(self) -> Dict[str, Any]:
        """Статус менеджера"""
        return {
            'model': self.model,
            'embedding_dim': self.embedding_dim,
            'status': 'active'
        }