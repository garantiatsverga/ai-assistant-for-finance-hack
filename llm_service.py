import logging
import time
from typing import List, Dict, Any
from openai import OpenAI

logger = logging.getLogger(__name__)

class LLMService:
    def __init__(self, api_key: str, base_url: str, model: str, headers: Dict = None):
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
                        wait_time = (attempt + 1) * 10
                        logger.info(f"Модель временно недоступна. Ждем {wait_time} сек перед повтором...")
                        time.sleep(wait_time)
                        continue
                    else:
                        logger.error("Превышено количество попыток из-за ошибки 429.")
                        return f"Ошибка генерации ответа: Модель временно недоступна (превышено количество попыток)."
                else:
                    return f"Ошибка при обращении к языковой модели: {str(e)}"

        return "Ошибка генерации ответа: Модель временно недоступна."

    def _format_context(self, context_docs: List[Dict[str, Any]]) -> str:
        """Форматирование контекста для промпта"""
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
        """Создание промпта для модели (я использовал Mistral)"""
        return f"""
Ты — финансовый помощник. Используй приведённый ниже контекст, чтобы кратко и точно ответить на вопрос. Общайся вежливо и пиши обширные ответы.

Контекст:
{context}

Вопрос:
{question}

Ответ:
"""