# rag_core.py
import numpy as np
from typing import List, Dict, Any, Optional
import logging
from embedding_service import *
from llm_service import *
from typing import List, Dict, Any
import logging
from openai import OpenAI
import os

logger = logging.getLogger(__name__)

class LLMService:
    def __init__(self, api_key: str, base_url: str, model: str, headers: Dict = None):
        self.client = OpenAI(
            base_url=base_url,
            api_key=api_key
        )
        self.model = model
        self.headers = headers
    
    def generate_answer(self, question: str, context_docs: List[Dict[str, Any]]) -> str:
        """Генерация ответа с контекстом для Mistral"""
        if not context_docs:
            return "В базе знаний нет релевантной информации для ответа на этот вопрос."
        
        # Формируем контекст из документов
        context_text = self._format_context(context_docs)
        prompt = self._create_mistral_prompt(question, context_text)
        
        try:
            # Дополнительные параметры для Mistral
            extra_params = {}
            if self.headers:
                extra_params["extra_headers"] = self.headers
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=1000,
                **extra_params
            )
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"Ошибка генерации ответа: {e}")
            return f"Ошибка при обращении к языковой модели: {str(e)}"
    
    def _format_context(self, context_docs: List[Dict[str, Any]]) -> str:
        """Форматирование контекста для промпта"""
        context_parts = []
        for i, doc_info in enumerate(context_docs):
            # Обрезаем длинные тексты
            text = doc_info['text']
            if len(text) > 1500:
                text = text[:1500] + "..."
                
            context_parts.append(f"Документ {i+1} (схожесть: {doc_info['similarity']:.3f}):\n{text}")
        return "\n\n".join(context_parts)
    
    def _create_mistral_prompt(self, question: str, context: str) -> str:
        """Специальный промпт для Mistral"""
        return f"""<s>[INST] Ты - финансовый консультант банка. Ответь на вопрос клиента, используя ТОЛЬКО информацию из предоставленных документов.

ВОПРОС КЛИЕНТА: {question}

ИНФОРМАЦИЯ ИЗ БАЗЫ ЗНАНИЙ:
{context}

ИНСТРУКЦИИ:
1. Ответь точно на вопрос используя только предоставленную информацию
2. Если информации недостаточно, честно скажи об этом
3. Не придумывай информацию, которой нет в документах
4. Будь профессиональным и полезным
5. Форматируй ответ четко и структурированно

ОТВЕТ: [/INST]"""
    
logger = logging.getLogger(__name__)

class RAGCore:
    """Ядро RAG системы - поиск и ранжирование документов"""
    
    def __init__(self, embedding_manager: EmbeddingManager):
        self.embedding_manager = embedding_manager
        self.documents = []
        self.document_embeddings = None
    
    def load_documents(self, documents: List[Dict[str, Any]]) -> bool:
        """Загрузка документов в систему"""
        try:
            self.documents = documents
            logger.info(f"Загружено {len(documents)} документов в RAG")
            return True
        except Exception as e:
            logger.error(f"Ошибка загрузки документов: {e}")
            return False
    
    def precompute_embeddings(self) -> bool:
        """Предварительное вычисление эмбеддингов для всех документов"""
        if not self.documents:
            logger.error("Нет документов для вычисления эмбеддингов")
            return False
            
        try:
            texts = [doc['text'] for doc in self.documents]
            self.document_embeddings = self.embedding_manager.encode_batch(texts)
            logger.info(f"Вычислены эмбеддинги для {len(self.documents)} документов")
            return True
        except Exception as e:
            logger.error(f"Ошибка вычисления эмбеддингов: {e}")
            return False
    
    def retrieve(self, question: str, top_k: int = 3, similarity_threshold: float = 0.3) -> List[Dict[str, Any]]:
        """Поиск релевантных документов"""
        if not self.document_embeddings:
            logger.error("Эмбеддинги документов не вычислены")
            return []
        
        try:
            # Получаем эмбеддинг вопроса
            question_embedding = self.embedding_manager.encode(question)
            if question_embedding is None:
                return []
            
            # Вычисляем схожесть с каждым документом
            similarities = []
            for doc_emb in self.document_embeddings:
                similarity = self._cosine_similarity(question_embedding, doc_emb)
                similarities.append(similarity)
            
            # Ранжируем и фильтруем по порогу
            results = []
            sorted_indices = np.argsort(similarities)[::-1]
            
            for idx in sorted_indices[:top_k]:
                if similarities[idx] >= similarity_threshold:
                    results.append({
                        'document': self.documents[idx],
                        'similarity': similarities[idx],
                        'text': self.documents[idx]['text']
                    })
            
            logger.info(f"Найдено {len(results)} релевантных документов")
            return results
            
        except Exception as e:
            logger.error(f"Ошибка поиска документов: {e}")
            return []
    
    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Вычисление косинусного сходства"""
        try:
            return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
        except Exception:
            return 0.0
    
    def get_status(self) -> Dict[str, Any]:
        """Статус системы"""
        return {
            'documents_loaded': len(self.documents),
            'embeddings_computed': self.document_embeddings is not None,
            'embedding_manager_status': self.embedding_manager.get_status()
        }