import os
import logging
import pandas as pd
import time
import json
import numpy as np
import requests
from typing import List, Dict, Any, Optional
from collections import Counter
from dotenv import load_dotenv
from openai import OpenAI
from time import sleep

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Загрузка переменных окружения
load_dotenv()

# embedding_service.py

class EmbeddingManager:
    """Менеджер для работы с эмбеддингами через API"""

    def __init__(self, api_key: str, base_url: str, model: str = "text-embedding-3-small", headers: dict = None, delay_per_request: float = 0.5):
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.model = model
        self.embedding_dim = 1536
        self.headers = headers
        self.delay_per_request = delay_per_request

    def encode(self, text: str) -> Optional[np.ndarray]:
        """Получение эмбеддинга для одного текста"""
        if self.delay_per_request > 0:
            time.sleep(self.delay_per_request)
        try:
            response = self.client.embeddings.create(model=self.model, input=[text[:8000]])
            return np.array(response.data[0].embedding)
        except Exception as e:
            logger.error(f"Ошибка получения эмбеддинга: {e}")
            return None

    def encode_batch(self, texts: List[str]) -> Optional[List[np.ndarray]]:
        """Получение эмбеддингов для батча текстов"""
        if not texts:
            return []
        if self.delay_per_request > 0:
            time.sleep(self.delay_per_request)
        try:
            truncated_texts = [text[:8000] for text in texts]
            response = self.client.embeddings.create(model=self.model, input=truncated_texts)
            embeddings = [np.array(item.embedding) for item in response.data]
            return embeddings
        except Exception as e:
            logger.error(f"Ошибка получения эмбеддингов батчом: {e}")
            return None

# full_text_search.py (BM25)

from tokenizers import (
    decoders,
    models,
    normalizers,
    pre_tokenizers,
    trainers,
    Tokenizer,
)

class FullTextSearch:
    """Реализация BM25 для полнотекстового поиска"""

    def __init__(self, documents: List[str], k1: float = 1.2, b: float = 0.75):
        self.documents = documents
        self.k1 = k1
        self.b = b
        self.N = len(documents)
        self.tokenizer = Tokenizer(models.WordPiece(unk_token="[UNK]"))
        self.train_tokenizer()
        self.token_frequency = self.count_frequency_tokens(documents)
        self.tokens_by_document = self.count_tokens_by_document(documents)
        self.avg_length = sum([t.total() for t in self.tokens_by_document]) / self.N

    def train_tokenizer(self):
        self.tokenizer.normalizer = normalizers.Sequence(
            [normalizers.NFD(), normalizers.Lowercase(), normalizers.StripAccents()]
        )
        self.tokenizer.pre_tokenizer = pre_tokenizers.BertPreTokenizer()
        special_tokens = ["[UNK]"]
        trainer = trainers.WordPieceTrainer(vocab_size=15000, special_tokens=special_tokens)
        self.tokenizer.train_from_iterator(self.get_training_corpus(), trainer=trainer)
        self.tokenizer.decoder = decoders.WordPiece(prefix="##")
        self.tokenizer.save("bm25tokenizer.json")

    def get_training_corpus(self):
        for i in range(0, len(self.documents), 1000):
            yield self.documents[i: i + 1000]

    def count_frequency_tokens(self, documents):
        counts = {}
        for doc in documents:
            for token_id in set(self.tokenizer.encode(doc).ids):
                counts[token_id] = counts.get(token_id, 0) + 1
        return counts

    def count_tokens_by_document(self, documents):
        counts = [Counter(self.tokenizer.encode(doc).ids) for doc in documents]
        return counts

    def idf(self, token_ids):
        counts = []
        for token_id in token_ids:
            freq = self.token_frequency.get(token_id, 0)
            counts.append(freq)
        counts = np.array(counts)
        return np.log((self.N - counts + 0.5) / (counts + 0.5) + 1)

    def token_frequency_by_document(self, token_ids):
        result = []
        for tokens in self.tokens_by_document:
            doc_length = tokens.total()
            counts = []
            for token_id in token_ids:
                freq = tokens.get(token_id, 0)
                counts.append(freq)
            counts = np.array(counts)
            denominator = counts + self.k1 * (1 - self.b + self.b * doc_length / self.avg_length)
            result.append(counts * (self.k1 + 1) / denominator)
        return np.array(result)

    def get_scores(self, query: str) -> np.ndarray:
        token_ids = self.tokenizer.encode(query).ids
        token_ids = [tid for tid in token_ids if tid != 0]
        if not token_ids:
            return np.zeros(self.N)
        token_freq = self.token_frequency_by_document(token_ids)
        idf = self.idf(token_ids)
        return token_freq @ idf

# llm_service.py

class LLMService:
    """Сервис для генерации ответов с помощью LLM"""

    def __init__(self, api_key: str, base_url: str, model: str, headers: Dict = None):
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.model = model
        self.headers = headers

    def generate_answer(self, question: str, context_docs: List[Dict[str, Any]]) -> str:
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
                        wait_time = (attempt + 1) * 10
                        logger.info(f"Модель временно недоступна. Ждем {wait_time} сек перед повтором...")
                        time.sleep(wait_time)
                        continue
                    else:
                        logger.error("Превышено количество попыток из-за ошибки 429.")
                        return "Ошибка генерации ответа: Модель временно недоступна (превышено количество попыток)."
                else:
                    return f"Ошибка при обращении к языковой модели: {str(e)}"
        return "Ошибка генерации ответа: Модель временно недоступна."

    def _format_context(self, context_docs: List[Dict[str, Any]]) -> str:
        context_parts = []
        for i, doc_info in enumerate(context_docs):
            text = doc_info['document'].get('original_text', '')
            if len(text) > 1500:
                text = text[:1500] + "..."
            tags = doc_info['document'].get('tags', [])
            tags_str = f" [Теги: {', '.join(tags)}]" if tags else ""
            score_info = f" (Rerank Score: {doc_info.get('relevance_score', 'N/A'):.3f})" if 'relevance_score' in doc_info else f" (BM25 Score: {doc_info.get('bm25_score', 'N/A'):.3f})"
            context_parts.append(f"Документ {i+1}{score_info}{tags_str}:\n{text}\n---\n")
        return "".join(context_parts)

    def _create_mistral_prompt(self, question: str, context: str) -> str:
        return f"""
Ты — финансовый помощник. Используй приведённый ниже контекст, чтобы кратко и точно ответить на вопрос. Общайся вежливо и пиши обширные ответы.
Контекст:
{context}
Вопрос:
{question}
Ответ:
"""

# rag_core.py

class RAGCore:
    """Ядро RAG системы — использует BM25 и опционально реранкер через API"""

    def __init__(self, embedding_manager: EmbeddingManager, reranker_api_key: Optional[str] = None):
        self.embedding_manager = embedding_manager
        self.reranker_api_key = reranker_api_key
        self.bm25 = None
        self.documents = []

    def load_documents(self, documents: List[Dict[str, Any]]) -> bool:
        try:
            self.documents = documents
            logger.info(f"Загружено {len(documents)} документов в RAG")
            return True
        except Exception as e:
            logger.error(f"Ошибка загрузки документов: {e}")
            return False

    def precompute_bm25(self) -> bool:
        try:
            texts = [doc['text'] for doc in self.documents]
            self.bm25 = FullTextSearch(texts, k1=1.2, b=0.75)
            logger.info("BM25 обучен")
            return True
        except Exception as e:
            logger.error(f"Ошибка обучения BM25: {e}")
            return False

    def _rerank_with_api(self, query: str, candidate_docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Выполняет реранкинг через официальный API реранкера"""
        if not self.reranker_api_key:
            logger.warning("Ключ реранкера не задан. Пропуск реранкинга.")
            return candidate_docs

        documents_text = [doc['document']['text'] for doc in candidate_docs]
        url = "https://ai-for-finance-hack.up.railway.app/rerank"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.reranker_api_key}"
        }
        payload = {
            "model": "deepinfra/Qwen/Qwen3-Reranker-4B",
            "query": query,
            "documents": documents_text
        }

        try:
            response = requests.post(url, headers=headers, json=payload, timeout=30)
            response.raise_for_status()
            result = response.json()

            for i, item in enumerate(result.get('results', [])):
                candidate_docs[i]['relevance_score'] = item['relevance_score']

            candidate_docs.sort(key=lambda x: x.get('relevance_score', -float('inf')), reverse=True)
            logger.info("Реранкинг через API завершён успешно")
            sleep(5)
            return candidate_docs

        except Exception as e:
            logger.error(f"Ошибка при вызове реранкера: {e}")
            logger.warning("Возвращаем результаты BM25 без реранкинга")
            return candidate_docs

    def retrieve(self, question: str, top_k: int = 25, use_bm25: bool = True, use_faiss: bool = False, use_reranker: bool = False) -> List[Dict[str, Any]]:
        if not use_bm25:
            logger.warning("FAISS отключён. BM25 обязателен.")
            return []

        if self.bm25 is None:
            logger.error("BM25 не обучен. Вызовите precompute_bm25().")
            return []

        scores = self.bm25.get_scores(question)
        top_indices = np.argsort(-scores)[:top_k]
        candidates = []
        for idx in top_indices:
            if scores[idx] <= 0:
                break
            candidates.append({
                'document': self.documents[idx],
                'bm25_score': float(scores[idx]),
                'text': self.documents[idx]['original_text'],
                'tags': self.documents[idx].get('tags', []),
            })

        if use_reranker and self.reranker_api_key:
            candidates = self._rerank_with_api(question, candidates)

        return candidates

# Вспомогательные функции загрузки

def load_train_documents(csv_path: str) -> List[Dict[str, Any]]:
    df = pd.read_csv(csv_path, on_bad_lines='skip', quoting=1)
    documents = []
    for _, row in df.iterrows():
        text_part = row.get('text', '')
        tags_part = ' '.join(row.get('tags', [])) if isinstance(row.get('tags'), list) else row.get('tags', '')
        combined_text = ' '.join([text_part, tags_part]).strip()
        documents.append({
            "text": combined_text,
            "original_text": row.get('text', ''),
            "tags": row.get('tags', []),
        })
    logger.info(f"Загружено {len(documents)} документов из {csv_path}")
    return documents

def load_questions(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    logger.info(f"Загружено {len(df)} вопросов")
    return df

# Основная функция

def main():
    BASE_URL = "https://ai-for-finance-hack.up.railway.app/"
    EMBEDDER_API_KEY = os.getenv("EMBEDDER_API_KEY")
    LLM_API_KEY = os.getenv("LLM_API_KEY")
    RERANKER_API_KEY = os.getenv("RERANKER_API_KEY")

    if not EMBEDDER_API_KEY or not LLM_API_KEY:
        raise ValueError("Отсутствуют EMBEDDER_API_KEY или LLM_API_KEY в .env файле!")

    embedding_manager = EmbeddingManager(
        api_key=EMBEDDER_API_KEY,
        base_url=BASE_URL,
        model="text-embedding-3-small",
        delay_per_request=0.5
    )

    llm_service = LLMService(
        api_key=LLM_API_KEY,
        base_url=BASE_URL,
        model="openrouter/mistralai/mistral-small-3.2-24b-instruct"
    )

    rag = RAGCore(embedding_manager=embedding_manager, reranker_api_key=RERANKER_API_KEY)

    documents = load_train_documents("train_data.csv")
    if not rag.load_documents(documents):
        logger.error("Не удалось загрузить документы")
        return

    if not rag.precompute_bm25():
        logger.error("Не удалось обучить BM25")
        return

    logger.info("RAG ядро (с BM25 и реранкером) готово")

    questions_df = load_questions("questions.csv")
    answers = []
    for _, row in questions_df.iterrows():
        q_id = row["ID вопроса"]
        question = row["Вопрос"]
        logger.info(f"Обработка вопроса {q_id}: {question[:60]}...")

        retrieved_docs = rag.retrieve(
            question,
            top_k=25,
            use_bm25=True,
            use_faiss=False,
            use_reranker=True
        )

        answer = llm_service.generate_answer(question, retrieved_docs)
        answers.append({"ID вопроса": q_id, "Ответ": answer})
        time.sleep(5)

    submission_df = pd.DataFrame(answers)
    submission_df.to_csv("submission.csv", index=False, quoting=1)
    logger.info("Файл submission.csv успешно сохранён!")

# Запуск

if __name__ == "__main__":
    main()