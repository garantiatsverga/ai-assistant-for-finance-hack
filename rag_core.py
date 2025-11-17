import numpy as np
import faiss
import requests
import logging
from typing import List, Dict, Any, Optional
from embedding_service import EmbeddingManager
from full_text_search import FullTextSearch

logger = logging.getLogger(__name__)

class RAGCore:
    """Ядро RAG системы - поиск и ранжирование документов с использованием BM25, FAISS и реранкера"""

    def __init__(self, embedding_manager: EmbeddingManager, reranker_api_key: str = None):
        self.embedding_manager = embedding_manager
        self.documents = []
        self.index = None
        self.embedding_dim = None
        self.reranker_api_key = reranker_api_key
        self.bm25_search = None

    def load_documents(self, documents: List[Dict[str, Any]]) -> bool:
        """Загрузка документов в систему"""
        try:
            self.documents = documents
            logger.info(f"Загружено {len(documents)} документов в RAG")
            return True
        except Exception as e:
            logger.error(f"Ошибка загрузки документов: {e}")
            return False

    def precompute_bm25(self, k1=1.2, b=0.75) -> bool:
        """Предварительное обучение BM25 индекса"""
        if not self.documents:
            logger.error("Нет документов для обучения BM25")
            return False
        try:
            # Используем 'text' для BM25
            texts = [doc['text'] for doc in self.documents]
            self.bm25_search = FullTextSearch(texts, k1, b)
            logger.info(f"BM25 индекс обучен на {len(self.documents)} документах")
            return True
        except Exception as e:
            logger.error(f"Ошибка при обучении BM25: {e}")
            return False

    def precompute_embeddings(self) -> bool:
        """Предварительное вычисление эмбеддингов и создание FAISS индекса (опционально)"""
        if not self.documents:
            logger.error("Нет документов для вычисления эмбеддингов")
            return False
        try:
            texts = [doc['text'] for doc in self.documents]
            embeddings = self.embedding_manager.encode_batch(texts)

            if embeddings is None:
                logger.error("Не удалось получить эмбеддинги")
                return False

            self.embedding_dim = len(embeddings[0])
            embeddings_matrix = np.array(embeddings).astype('float32')
            faiss.normalize_L2(embeddings_matrix)

            self.index = faiss.IndexFlatIP(self.embedding_dim)
            self.index.add(embeddings_matrix)

            logger.info(f"FAISS индекс создан для {len(self.documents)} документов")
            return True
        except Exception as e:
            logger.error(f"Ошибка при создании FAISS индекса: {e}")
            return False

    def rerank_documents(self, query: str, documents: List[str]) -> List[Dict[str, Any]]:
        """Rerank документов с помощью внешней модели"""
        if not self.reranker_api_key:
            logger.warning("API ключа реранкера не установлен. Возвращаем документы без реранка.")
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
            logger.error(f"Ошибка при реранке: {e}")
            return [{"document_text": d, "relevance_score": 1.0} for d in documents]

    def retrieve(self, question: str, top_k: int = 25, use_bm25=True, use_faiss=False, use_reranker=True) -> List[Dict[str, Any]]:
        """Поиск релевантных документов с использованием BM25, FAISS и/или реранкера"""
        if use_bm25 and self.bm25_search:
            # BM25 поиск
            bm25_scores = self.bm25_search.get_scores(question)
            top_bm25_indices = np.argsort(bm25_scores)[::-1][:top_k*3]
            initial_results = []
            for idx in top_bm25_indices:
                if idx < len(self.documents):
                    initial_results.append({
                        'document': self.documents[idx],
                        'bm25_score': bm25_scores[idx],
                        'text': self.documents[idx]['original_text'],
                        'tags': self.documents[idx].get('tags', []),
                    })
            logger.info(f"BM25 нашёл {len(initial_results)} кандидатов.")
        elif use_faiss and self.index:
            # FAISS поиск
            question_embedding = self.embedding_manager.encode(question)
            if question_embedding is None:
                logger.error("Не удалось получить эмбеддинг вопроса для FAISS")
                return []
            query_vec = question_embedding.astype('float32').reshape(1, -1)
            faiss.normalize_L2(query_vec)
            similarities, indices = self.index.search(query_vec, top_k*3)
            initial_results = []
            for sim, idx in zip(similarities[0], indices[0]):
                if idx < len(self.documents):
                    initial_results.append({
                        'document': self.documents[idx],
                        'similarity': float(sim),
                        'text': self.documents[idx]['original_text'],
                        'tags': self.documents[idx].get('tags', []),
                    })
            logger.info(f"FAISS нашёл {len(initial_results)} кандидатов.")
        else:
            logger.error("Ни BM25, ни FAISS не инициализированы или не выбраны.")
            return []

        if not initial_results:
            logger.info("Нет кандидатов для реранка.")
            return []

        if use_reranker and self.reranker_api_key:
            candidate_texts = [res['text'] for res in initial_results]
            reranked_results = self.rerank_documents(question, candidate_texts)

            final_results = []
            for rr in reranked_results:
                for ir in initial_results:
                    if ir['text'] == rr['document_text']:
                        enriched_result = {
                            'document': ir['document'],
                            'relevance_score': rr['relevance_score'],
                            'text': ir['text'],
                            'tags': ir['tags'],
                        }
                        if 'bm25_score' in ir:
                            enriched_result['bm25_score'] = ir['bm25_score']
                        if 'similarity' in ir:
                            enriched_result['similarity'] = ir['similarity']
                        final_results.append(enriched_result)
                        break
                if len(final_results) >= top_k:
                    break
            final_results = final_results[:top_k]
            logger.info(f"Rerank завершён, возвращено {len(final_results)} документов.")
            return final_results
        # Если реранкер не используется, возвращаем топ-K из initial_results, отсортированных по bm25 или similarity
        sorted_initial = sorted(initial_results, key=lambda x: x.get('bm25_score', x.get('similarity', 0)), reverse=True)
        return sorted_initial[:top_k]


    def get_status(self) -> Dict[str, Any]:
        """Статус RAG ядра"""
        return {
            'documents_loaded': len(self.documents),
            'bm25_built': self.bm25_search is not None,
            'faiss_index_built': self.index is not None,
            'embedding_dim': self.embedding_dim,
            'embedding_manager_status': self.embedding_manager.get_status()
        }