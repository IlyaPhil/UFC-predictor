import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import aiohttp
import streamlit as st
import json
from aiohttp import MultipartWriter

# base_url = "http://0.0.0.0:8000/api"
base_url = "https://ufc-predictor.streamlit.app"

# Создайте функцию для загрузки данных в endpoint
async def upload_json_data_to_endpoint(file):
    url = base_url + "/upload_fight_data_json/"  
    async with aiohttp.ClientSession() as session:
        async with session.post(url, json=json.load(file)) as response:
            if response.status == 200:
                return await response.json()
            else:
                st.error("Ошибка при загрузке данных")
                return None


# Создайте функцию для загрузки данных в endpoint
async def upload_csv_data_to_endpoint(file):
    url = base_url + "/upload_fight_data_csv/"
    async with aiohttp.ClientSession() as session:
        async with session.post(url, data={'file': file}) as response:
            if response.status == 200:
                return await response.json()
            else:
                st.error("Ошибка при загрузке данных")
                return None
            

# Создайте функцию для вызова endpoint
async def select_model_endpoint(model_name):
    url = base_url + f"/select-model/{model_name}"
    async with aiohttp.ClientSession() as session:
        async with session.post(url) as response:
            if response.status == 200:
                return await response.json()
            else:
                st.error("Ошибка при выборе модели")
                return None
            

# Создайте функцию для вызова endpoint
async def get_predictions(bet_size):
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
    ax.hist(y_proba[:, 0], alpha=0.5, label='Red Fighter')
    ax.hist(y_proba[:, 1], alpha=0.5, label='Blue Fighter')
    ax.set_xlabel('Вероятность')
    ax.set_ylabel('Частота')
    ax.set_title('график уверенности модели при выборе победителя')
    ax.legend()
    return fig
    