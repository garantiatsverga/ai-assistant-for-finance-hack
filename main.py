import os
import logging
import pandas as pd
import time
from typing import List, Dict, Any

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Загрузка переменных окружения
from dotenv import load_dotenv
load_dotenv()

# Импорты моих модулей
from embedding_service import EmbeddingManager
from rag_core import LLMService, RAGCore

def load_train_documents(csv_path: str) -> List[Dict[str, Any]]:
    """Загрузка документов из train_data.csv с объединением всех релевантных полей"""
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
    """Загрузка вопросов из questions.csv"""
    df = pd.read_csv(csv_path)
    logger.info(f"Загружено {len(df)} вопросов")
    return df

def main():
    # Конфигурация
    BASE_URL = "https://ai-for-finance-hack.up.railway.app/"

    # Ключи из .env
    EMBEDDER_API_KEY = os.getenv("EMBEDDER_API_KEY")
    LLM_API_KEY = os.getenv("LLM_API_KEY")
    RERANKER_API_KEY = os.getenv("EMBEDDER_API_KEY")

    if not EMBEDDER_API_KEY or not RERANKER_API_KEY:
        raise ValueError("Отсутствует ключ эмбеддера .env файле!")
    if not LLM_API_KEY:
        raise ValueError("Отсутствует API-ключ в .env-файле!")

    # Инициализация сервисов
    embedding_manager = EmbeddingManager(
        api_key=EMBEDDER_API_KEY,
        base_url=BASE_URL,
        model="text-embedding-3-small",
        delay_per_request=0.2
    )

    llm_service = LLMService(
        api_key=LLM_API_KEY,
        base_url=BASE_URL,
        model="openrouter/mistralai/mistral-small-3.2-24b-instruct"
    )

    # Передаём RERANKER_API_KEY
    rag = RAGCore(embedding_manager=embedding_manager, reranker_api_key=RERANKER_API_KEY)

    # Загрузка и индексация документов
    documents = load_train_documents("train_data.csv")
    if not rag.load_documents(documents):
        logger.error("Не удалось загрузить документы")
        return

    if not rag.precompute_embeddings():
        logger.error("Не удалось вычислить эмбеддинги")
        return

    logger.info("RAG ядро готово")

    # Обработка вопросов
    questions_df = load_questions("questions.csv")
    answers = []

    for _, row in questions_df.iterrows():
        q_id = row["ID вопроса"]
        question = row["Вопрос"]

        logger.info(f"Обработка вопроса {q_id}: {question[:60]}...")

        retrieved_docs = rag.retrieve(
            question,
            top_k=5,
            similarity_threshold=0.05,
            use_reranker=True
        )
        # ---
        answer = llm_service.generate_answer(question, retrieved_docs)
        answers.append({"ID вопроса": q_id, "Ответ": answer})

        # Задержка между запросами
        time.sleep(5)

    # Сохранение результата
    submission_df = pd.DataFrame(answers)
    submission_df.to_csv("submission.csv", index=False, quoting=1)
    logger.info("Файл submission.csv успешно сохранён!")
    logger.info(f"Пример ответа:\nID: {answers[0]['ID вопроса']}\nОтвет: {answers[0]['Ответ'][:200]}...")

if __name__ == "__main__":
    main()