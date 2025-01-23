import streamlit as st
import pandas as pd
import aiohttp
import json
from st_functions import upload_json_data_to_endpoint

st.title('UFC fight winner predictor')

st.header('Загрузка данных')

# Создайте виджет для загрузки файла JSON
st.title("Загрузка данных о бое")
file = st.file_uploader("Выберите файл JSON", type=["json"])

if file is not None:
    # Загрузите данные в endpoint
    import asyncio
    loop = asyncio.get_event_loop()
    response = loop.run_until_complete(upload_json_data_to_endpoint(file))
    if response is not None:
        # Отобразите датафрейм на экране
        df = pd.DataFrame(response["dataframe"])
        st.write("Загруженные данные:")
        st.dataframe(df)

# Добавляем загрузку данных
# uploaded_file = st.file_uploader("Выберите CSV-файл", type=["csv"])