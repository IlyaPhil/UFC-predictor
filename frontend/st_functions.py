"""
Импортируемые зависимости
"""
import json
import matplotlib.pyplot as plt
import aiohttp
import streamlit as st


base_url = "http://0.0.0.0:8000/api"


async def upload_json_data_to_endpoint(file):
    """
    Загружает данных в формате JSON при помощи endpoint upload_fight_data_json
    """
    url = base_url + "/upload_fight_data_json/"
    async with aiohttp.ClientSession() as session:
        async with session.post(url, json=json.load(file)) as response:
            if response.status == 200:
                return await response.json()
            else:
                st.error("Ошибка при загрузке данных")
                return None



async def upload_csv_data_to_endpoint(file):
    """    
    Загружает данных в формате csv при помощи endpoint upload_fight_data_csv
    """
    url = base_url + "/upload_fight_data_csv/"
    async with aiohttp.ClientSession() as session:
        async with session.post(url, data={'file': file}) as response:
            if response.status == 200:
                return await response.json()
            else:
                st.error("Ошибка при загрузке данных")
                return None



async def select_model_endpoint(model_name):
    """    
    Выбор одной из предзагруженных моделей при помощи endpoint select_model
    """
    url = base_url + f"/select-model/{model_name}"
    async with aiohttp.ClientSession() as session:
        async with session.post(url) as response:
            if response.status == 200:
                return await response.json()
            else:
                st.error("Ошибка при выборе модели")
                return None



async def get_predictions(bet_size):
    """
    Вызов предсказания выбранной модели при помощи endpoint predict
    """
    url = base_url + "/predict"
    payload = {"bet_size": bet_size}
    async with aiohttp.ClientSession() as session:
        async with session.post(url, json=payload) as response:
            if response.status == 200:
                return await response.json()
            else:
                st.error("Ошибка при получении предсказаний")
                return None


def plot_probabilities(y_proba: list) -> plt.figure:
    """
    Построение графика уверенности модели при выборе победителя
    """
    fig, ax = plt.subplots()
    ax.hist(y_proba[:, 1], alpha=0.5, label='Blue Fighter')
    ax.hist(y_proba[:, 0], alpha=0.5, label='Red Fighter')
    ax.set_xlabel('Вероятность')
    ax.set_ylabel('Частота')
    ax.set_title('График уверенности модели при выборе победителя')
    ax.legend()
    return fig
