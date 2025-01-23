import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import aiohttp
import streamlit as st
import json

base_url = 'http://localhost:8501'

# Создайте функцию для загрузки данных в endpoint
async def upload_json_data_to_endpoint(file):
    url = "base_url" + "/upload_fight_data_json/"  
    async with aiohttp.ClientSession() as session:
        async with session.post(url, json=json.load(file)) as response:
            if response.status == 200:
                return await response.json()
            else:
                st.error("Ошибка при загрузке данных")
                return None