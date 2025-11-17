import os
import logging
import pandas as pd
import time
from typing import List, Dict, Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from dotenv import load_dotenv
load_dotenv()

from embedding_service import EmbeddingManager
from rag_core import RAGCore
from llm_service import LLMService

def load_train_documents(csv_path: str) -> List[Dict[str, Any]]:
    """Загрузка документов из train_data.csv с объединением всех релевантных полей"""
    df = pd.read_csv(csv_path, on_bad_lines='skip', quoting=1)
    documents = []
    for _, row in df.iterrows():
        text_part = row.get('text', '')
        tags_part = ' '.join(row.get('tags', [])) if isinstance(row.get('tags'), list) else row.get('tags', '')
        combined_text = ' '.join([text_part, tags_part]).strip()

        documents.append({
            "text": combined_text,          # Для BM25/FAISS
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
    BASE_URL = "https://ai-for-finance-hack.up.railway.app/"

    EMBEDDER_API_KEY = os.getenv("EMBEDDER_API_KEY")
    LLM_API_KEY = os.getenv("LLM_API_KEY")
    RERANKER_API_KEY = os.getenv("RERANKER_API_KEY")

    if not EMBEDDER_API_KEY or not LLM_API_KEY:
        raise ValueError("Отсутствуют EMBEDDER_API_KEY или LLM_API_KEY в .env-файле!")

    embedding_manager = EmbeddingManager(
        api_key=EMBEDDER_API_KEY,
        base_url=BASE_URL,
        model="text-embedding-3-small",
        delay_per_request=0.5 # Задержка между запросами к эмбеддингам (вопрос, реранк)
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

    # Не вызываем precompute_embeddings, чтобы не расходовать квоту

    logger.info("RAG ядро (с BM25) готово")

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
            use_faiss=False, # Отключено
            use_reranker=True # Реранкер использует эмбеддинги вопроса и кандидатов
        )
        answer = llm_service.generate_answer(question, retrieved_docs)
        answers.append({"ID вопроса": q_id, "Ответ": answer})

        time.sleep(5) # Задержка между вопросами

    submission_df = pd.DataFrame(answers)
    submission_df.to_csv("submission.csv", index=False, quoting=1)
    logger.info("Файл submission.csv успешно сохранён!")

if __name__ == "__main__":
    main()