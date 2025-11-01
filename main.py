# solution.py
import pandas as pd
from tqdm import tqdm
import os
from dotenv import load_dotenv
import logging

# ЯВНО загружаем .env в начале
load_dotenv()

from rag_core import RAGCore
from embedding_service import EmbeddingManager
from llm_service import LLMService

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HackathonSolution:
    def __init__(self):
        api_key = os.getenv("EMBEDDER_API_KEY")
        if not api_key:
            raise ValueError("API-ключ не найден")
        
        # Инициализируем менеджеры
        self.embedding_manager = EmbeddingManager(
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1",
            model="text-embedding-3-small",
            headers={
                "HTTP-Referer": "https://finance-hackathon.com",
                "X-Title": "Finance Assistant"
            }
        )
        
        self.llm_service = LLMService(
            api_key=api_key,  # тот же ключ
            base_url="https://openrouter.ai/api/v1", 
            model="mistralai/mistral-7b-instruct",
            headers={
                "HTTP-Referer": "https://finance-hackathon.com",
                "X-Title": "Finance Assistant"
            }
        )
        
        self.rag_core = RAGCore(self.embedding_manager)
    
    def load_data(self, train_data_path: str) -> bool:
        """Загрузка тренировочных данных"""
        try:
            print("Загружаем данные...")
            df = pd.read_csv(train_data_path)
            documents = [
                {'id': row['id'], 'text': row['text']}
                for _, row in df.iterrows()
            ]
            
            print(f"Загружено {len(documents)} документов")
            
            if not self.rag_core.load_documents(documents):
                return False
                
            print("Вычисляем эмбеддинги...")
            return self.rag_core.precompute_embeddings()
            
        except Exception as e:
            logger.error(f"Ошибка загрузки данных: {e}")
            return False
    
    def process_questions(self, questions_path: str) -> pd.DataFrame:
        """Обработка вопросов"""
        try:
            questions_df = pd.read_csv(questions_path)
            questions_list = questions_df['Вопрос'].tolist()
            
            answers = []
            for i, question in enumerate(tqdm(questions_list, desc="Генерация ответов")):
                print(f"\nВопрос {i+1}/{len(questions_list)}: {question[:50]}...")
                
                # Поиск релевантных документов
                context_docs = self.rag_core.retrieve(question)
                print(f"Найдено документов: {len(context_docs)}")
                
                # Генерация ответа
                answer = self.llm_service.generate_answer(question, context_docs)
                answers.append(answer)
                print(f"Ответ сгенерирован")
            
            questions_df['Ответы на вопрос'] = answers
            return questions_df
            
        except Exception as e:
            logger.error(f"Ошибка обработки вопросов: {e}")
            raise

def main():
    """Главная функция"""
    try:
        print("Запуск RAG системы...")
        
        solution = HackathonSolution()
        
        # Загрузка данных
        if not solution.load_data('./train_data.csv'):
            print("Ошибка загрузки тренировочных данных")
            return
        
        # Обработка вопросов
        print("\nНачинаем обработку вопросов...")
        result_df = solution.process_questions('./questions.csv')
        
        # Сохранение результатов
        result_df.to_csv('submission.csv', index=False)
        print("\nРезультаты сохранены в submission.csv")
        
        # Показываем пример
        print("\nПример ответа:")
        print(result_df[['Вопрос', 'Ответы на вопрос']].head(1).to_string(index=False))
        
    except Exception as e:
        print(f"Критическая ошибка: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()