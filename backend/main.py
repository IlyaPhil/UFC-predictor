"""
Импортируемые модули
"""

import os
import sys
from contextlib import asynccontextmanager
import uvicorn
from fastapi import FastAPI
from api import router # Импортируем роутер из api_route.py
from models_store import models
import joblib
from models import RNNModel


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Загрузка моделей при старте приложения
    """
    print("Lifespan is being executed!")  # Тестовый вывод
    global models

    try:
        # Определяем директорию, в которой находится файл main.py

        current_dir = os.path.dirname(__file__)  # Получаем путь к текущему файлу
        model_path = os.path.join(current_dir, "../models/xgboost_model.joblib")
        models['XGBoost']= joblib.load(os.path.abspath(model_path))

        model_path = os.path.join(current_dir, "../models/rnn_model.joblib")
        models['RNN'] = joblib.load(os.path.abspath(model_path))

        print("Models loaded successfully!")
        print(f"Loaded models: {models.keys()}")  # Выводим ключи загруженных моделей
    except Exception as e:
        print(f"Models loading error: {e}")

    yield

app = FastAPI(
    title="UFC_predictor",
    docs_url="/api/openapi",
    openapi_url="/api/openapi.json",
    lifespan=lifespan)  # Подключаем lifespan

@app.get("/")
async def root():
    """
    Проверка статуса работы приложения
    """
    return {"status": "App healthy"}

# Подключаем роутер с префиксом /api
app.include_router(router, prefix="/api")

# Задаем автоматический перезапуск сервера после сохранения
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
