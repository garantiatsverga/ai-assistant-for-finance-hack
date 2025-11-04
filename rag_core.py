import numpy as np
import faiss
import requests
from typing import List, Dict, Any, Optional
import logging
from embedding_service import EmbeddingManager

logger = logging.getLogger(__name__)

class LLMService:
    def __init__(self, api_key: str, base_url: str, model: str, headers: Dict = None):
        from openai import OpenAI
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.model = model
        self.headers = headers

    def generate_answer(self, question: str, context_docs: List[Dict[str, Any]]) -> str:
        """Генерация ответа с контекстом"""
        if not context_docs:
            return "В базе знаний нет релевантной информации для ответа на этот вопрос."

        context_text = self._format_context(context_docs)
        prompt = self._create_mistral_prompt(question, context_text)

        max_retries = 3
        for attempt in range(max_retries):
            try:
                extra_params = {}
                if self.headers:
                    extra_params["extra_headers"] = self.headers

                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.1,
                    max_tokens=2000,
                    **extra_params
                )
                return response.choices[0].message.content
            except Exception as e:
                error_msg = str(e)
                logger.error(f"Ошибка генерации ответа (попытка {attempt + 1}): {e}")

                if "429" in error_msg or "cooldown" in error_msg.lower() or "No deployments available" in error_msg:
                    if attempt < max_retries - 1:
                        wait_time = (attempt + 1) * 10  # 10, 20, 30 сек
                        logger.info(f"Модель временно недоступна. Ждем {wait_time} сек перед повтором...")
                        import time
                        time.sleep(wait_time)
                        continue
                    else:
                        logger.error("Превышено количество попыток из-за ошибки 429.")
                        return f"Ошибка генерации ответа: Модель временно недоступна (превышено количество попыток)."
                else:
                    # Не 429 — возвращаем ошибку
                    return f"Ошибка при обращении к языковой модели: {str(e)}"

        # Если все попытки исчерпаны
        return "Ошибка генерации ответа: Модель временно недоступна."

    def _format_context(self, context_docs: List[Dict[str, Any]]) -> str:
        """Форматирование контекста для промпта"""
        context_parts = []
        for i, doc_info in enumerate(context_docs):
            # Используем original_text для предоставления чистой информации
            text = doc_info['document'].get('original_text', '')
            if len(text) > 1500:
                text = text[:1500] + "..."

            # Включаем теги, если они есть
            tags = doc_info['document'].get('tags', [])
            tags_str = f" [Теги: {', '.join(tags)}]" if tags else ""

            # Формируем строку без воды
            score_info = f" (Rerank Score: {doc_info.get('relevance_score', 'N/A'):.3f})" if 'relevance_score' in doc_info else f" (FAISS Sim: {doc_info['similarity']:.3f})"
            context_parts.append(f"Документ {i+1}{score_info}{tags_str}:\n{text}\n---\n")
        return "".join(context_parts)

    def _create_mistral_prompt(self, question: str, context: str) -> str:
        """Создание промпта для Mistral (или другой модели)"""
        return f"""
Ты — финансовый помощник. Используй приведённый ниже контекст, чтобы кратко и точно ответить на вопрос. Отвечай только на основе информации в контексте. Не придумывай ничего. Если в контексте нет информации, скажи: "В базе знаний нет релевантной информации для ответа на этот вопрос."

Контекст:
{context}

Вопрос:
{question}

Ответ:
"""


class RAGCore:
    """Ядро RAG системы - поиск и ранжирование документов с использованием FAISS и Reranker"""

    def __init__(self, embedding_manager: EmbeddingManager, reranker_api_key: str = None):
        self.embedding_manager = embedding_manager
        self.documents = []
        self.index = None  # FAISS индекс
        self.embedding_dim = None
        self.reranker_api_key = reranker_api_key


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
        """Предварительное вычисление эмбеддингов и создание FAISS индекса"""
        if not self.documents:
            logger.error("Нет документов для вычисления эмбеддингов")
            return False
        try:
            # Извлекаем тексты для эмбеддингов (combined_text)
            texts = [doc['text'] for doc in self.documents]
            embeddings = self.embedding_manager.encode_batch(texts)

            if embeddings is None:
                logger.error("Не удалось получить эмбеддинги")
                return False

            # Определяем размерность эмбеддингов (берём из первого)
            self.embedding_dim = len(embeddings[0])

            # Преобразуем в numpy массив (n_docs, dim)
            embeddings_matrix = np.array(embeddings).astype('float32')

            # Обязательная нормализация для косинусного сходства
            faiss.normalize_L2(embeddings_matrix)

            # Создаём FAISS индекс типа Flat (точный поиск)
            self.index = faiss.IndexFlatIP(self.embedding_dim)
            self.index.add(embeddings_matrix)

            logger.info(f"FAISS индекс создан для {len(self.documents)} документов")
            return True
        except Exception as e:
            logger.error(f"Ошибка при создании FAISS индекса: {e}")
            return False

    def rerank_documents(self, query: str, documents: List[str]) -> List[Dict[str, Any]]:
        """Реранк документов с помощью внешней модели"""
        if not self.reranker_api_key:
            logger.warning("Reranker API ключ не установлен. Возвращаем документы без реранка.")
            # Возвращаем в том же порядке с фиктивным скором
            return [{"document_text": d, "relevance_score": 1.0} for d in documents]

        url = "https://ai-for-finance-hack.up.railway.app/rerank"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.reranker_api_key}"
        }
        payload = {
            "model": "deepinfra/Qwen/Qwen3-Reranker-4B",
            "query": query,
            "documents": documents
        }

        try:
            response = requests.post(url, headers=headers, json=payload)
            response.raise_for_status()
            result = response.json()

            # Возвращаем результаты с relevance_score
            reranked_results = []
            for item in result.get('results', []):
                original_index = item['index']
                score = item['relevance_score']
                text = documents[original_index]
                reranked_results.append({
                    "document_text": text,
                    "relevance_score": score
                })
            logger.info(f"Реранк завершён для {len(documents)} документов.")
            return reranked_results
        except Exception as e:
            logger.error(f"Ошибка при rerank'е: {e}")
            # В случае ошибки возвращаем фиктивный счет
            return [{"document_text": d, "relevance_score": 1.0} for d in documents]

    def retrieve(self, question: str, top_k: int = 5, similarity_threshold: float = 0.05, use_reranker: bool = True) -> List[Dict[str, Any]]:
        """Поиск релевантных документов с помощью FAISS и опциональным реранком"""
        if self.index is None:
            logger.error("FAISS индекс не создан. Вызовите precompute_embeddings().")
            return []

        try:
            # Получаем эмбеддинг вопроса
            question_embedding = self.embedding_manager.encode(question)
            if question_embedding is None:
                logger.error("Не удалось получить эмбеддинг вопроса")
                return []

            # Подготавливаем эмбеддинг запроса
            query_vec = question_embedding.astype('float32').reshape(1, -1)
            faiss.normalize_L2(query_vec)

            # Поиск в FAISS (берём больше, чем нужно, для реранка)
            rerank_k = top_k * 3  # например, если top_k=5, ищем 15
            similarities, indices = self.index.search(query_vec, rerank_k)

            # Отфильтруем по порогу на уровне FAISS
            initial_results = []
            for sim, idx in zip(similarities[0], indices[0]):
                if sim >= similarity_threshold:
                    initial_results.append({
                        'document': self.documents[idx],
                        'similarity': float(sim), # схожесть из FAISS
                        'text': self.documents[idx]['original_text'],
                        'tags': self.documents[idx].get('tags', []),
                    })

            if not initial_results:
                logger.info("Нет документов, удовлетворяющих порогу схожести.")
                return []

            logger.info(f"Найдено {len(initial_results)} кандидатов до реранка.")

            if use_reranker:
                # Подготовим тексты для реранка
                candidate_texts = [res['text'] for res in initial_results]
                # Вызов реранкера
                reranked_results = self.rerank_documents(question, candidate_texts)

                final_results = []
                for rr in reranked_results:
                    # Найдём соответствующий документ в initial_results
                    for ir in initial_results:
                        if ir['text'] == rr['document_text']:
                            # Обогащаем результат rerank скором
                            final_results.append({
                                'document': ir['document'],
                                'similarity': ir['similarity'],  # схожесть из FAISS
                                'relevance_score': rr['relevance_score'], # счет из реранка
                                'text': ir['text'],
                                'tags': ir['tags'],
                            })
                            break  # нашли, выходим из внутреннего цикла
                    if len(final_results) >= top_k:  # ограничиваем количество
                        break

                final_results = final_results[:top_k]

                logger.info(f"Реранк завершён, возвращено {len(final_results)} документов.")
                return final_results
            else:
                # Если реранк не используется, просто возвращаем топ-K из initial_results
                return initial_results[:top_k]

        except Exception as e:
            logger.error(f"Ошибка поиска в RAG: {e}")
            return []

    def get_status(self) -> Dict[str, Any]:
        """Статус RAG ядра"""
        return {
            'documents_loaded': len(self.documents),
            'faiss_index_built': self.index is not None,
            'embedding_dim': self.embedding_dim,
            'embedding_manager_status': self.embedding_manager.get_status()
        }