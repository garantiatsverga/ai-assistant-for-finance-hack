# llm_service.py
from typing import List, Dict, Any
import logging
from openai import OpenAI
import os

logger = logging.getLogger(__name__)

class LLMService:
    def __init__(self, api_key: str, base_url: str, model: str, headers: Dict = None):
        """Инициализация сервиса LLM"""
        self.client = OpenAI(
            base_url=base_url,
            api_key=api_key
        )
        self.model = model
        self.headers = headers
        logger.info(f"LLMService инициализирован с моделью {model}")
    
    def generate_answer(self, question: str, context_docs: List[Dict[str, Any]]) -> str:
        """Генерация ответа с контекстом"""
        if not context_docs:
            return "В базе знаний нет релевантной информации для ответа на этот вопрос."
        
        # Формируем контекст из документов
        context_text = self._format_context(context_docs)
        prompt = self._create_prompt(question, context_text)
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=1000
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Ошибка генерации ответа: {e}")
            return f"Ошибка при обращении к языковой модели: {str(e)}"
    
    def _format_context(self, context_docs: List[Dict[str, Any]]) -> str:
        """Форматирование контекста для промпта"""
        context_parts = []
        for i, doc_info in enumerate(context_docs):
            context_parts.append(f"Документ {i+1} (схожесть: {doc_info['similarity']:.3f}):\n{doc_info['text']}")
        return "\n\n".join(context_parts)
    
    def _create_prompt(self, question: str, context: str) -> str:
        """Создание промпта"""
        return f"""Вопрос: {question}

Информация из базы знаний:
{context}

Инструкции: Ответь на вопрос используя только предоставленную информацию. Если информации недостаточно, укажи это."""