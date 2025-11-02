import faiss
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
        self.index = None
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
        if not self.documents:
            logger.error("Нет документов для вычисления эмбеддингов")
            return False
        try:
            texts = [doc['text'] for doc in self.documents]
            embeddings = self.embedding_manager.encode_batch(texts)
            if embeddings is None:
                logger.error("Не удалось получить эмбеддинги")
                return False

            # Определяем размерность эмбеддингов (берём из первого)
            self.embedding_dim = len(embeddings[0])

            # Преобразуем в numpy массив (n_docs, dim)
            embeddings_matrix = np.array(embeddings).astype('float32')

            # Нормализуем для использования косинусного сходства через L2
            faiss.normalize_L2(embeddings_matrix)

            # Создаём FAISS индекс типа Flat (точный поиск)
            self.index = faiss.IndexFlatIP(self.embedding_dim)  # Inner Product = Cosine после нормализации
            self.index.add(embeddings_matrix)

            logger.info(f"FAISS индекс создан для {len(self.documents)} документов")
            return True
        except Exception as e:
            logger.error(f"Ошибка при создании FAISS индекса: {e}")
            return False

    def retrieve(self, question: str, top_k: int = 3, similarity_threshold: float = 0.3) -> List[Dict[str, Any]]:
        if self.index is None:
            logger.error("FAISS индекс не создан. Вызовите precompute_embeddings().")
            return []

        try:
            question_embedding = self.embedding_manager.encode(question)
            if question_embedding is None:
                return []

            # Подготавливаем эмбеддинг запроса
            query_vec = question_embedding.astype('float32').reshape(1, -1)
            faiss.normalize_L2(query_vec)

            # Поиск в FAISS
            similarities, indices = self.index.search(query_vec, top_k)

            results = []
            for sim, idx in zip(similarities[0], indices[0]):
                if sim >= similarity_threshold:
                    results.append({
                        'document': self.documents[idx],
                        'similarity': float(sim),
                        'text': self.documents[idx]['text']
                    })

            logger.info(f"Найдено {len(results)} релевантных документов")
            return results
        except Exception as e:
            logger.error(f"Ошибка поиска в FAISS: {e}")
            return []
    
    def get_status(self) -> Dict[str, Any]:
        return {
            'documents_loaded': len(self.documents),
            'faiss_index_built': self.index is not None,
            'embedding_dim': self.embedding_dim,
            'embedding_manager_status': self.embedding_manager.get_status()
        }