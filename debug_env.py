import os
from dotenv import load_dotenv

load_dotenv()

print("🔍 Проверяем переменные окружения:")
print(f"EMBEDDER_API_KEY: {'***' + os.getenv('EMBEDDER_API_KEY')[-8:] if os.getenv('EMBEDDER_API_KEY') else 'НЕ НАЙДЕН'}")
print(f"Текущая директория: {os.getcwd()}")
print(f"Файлы в директории: {os.listdir('.')}")