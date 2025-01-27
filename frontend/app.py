"""
Импортируемые модули
"""
import asyncio
import streamlit as st
import pandas as pd
import numpy as np
from st_functions import upload_json_data_to_endpoint,\
                         upload_csv_data_to_endpoint,\
                         select_model_endpoint,\
                         get_predictions,\
                         plot_probabilities


st.title('UFC fight winner predictor')

st.header('Загрузка данных в json/csv')

# Выбор типа файла
file_type = st.selectbox("Выберите тип файла:", ["csv", "json"])

# Загрузка файла
if file_type == 'json':

    # Виджет для загрузки файла JSON
    file_json = st.file_uploader("Выберите файл JSON", type=["json"])

    if file_json is not None:
        # Загрузка данных в endpoint
        response = asyncio.run(upload_json_data_to_endpoint(file_json))
        if response is not None:
            # Отображение датафрейма на экране
            df = pd.DataFrame(response["dataframe"])
            st.write("Загруженные данные:")
            st.dataframe(df)

else:
    # Виджет для загрузки файла CSV
    file_csv = st.file_uploader("Выберите файл CSV", type=["csv"])

    if file_csv is not None:
        # Загрузите данные в endpoint
        response = asyncio.run(upload_csv_data_to_endpoint(file_csv))
        if response is not None:
            # Отобразите датафрейм на экране
            df = pd.DataFrame(response["dataframe"])
            st.write("Загруженные данные:")
            st.dataframe(df)


# Список доступных моделей
models = ["XGBoost", "RNN"]

selected_model = st.selectbox("Выберите модель", models)

# Вызов endpoint при изменении выбора модели
if selected_model:
    response = asyncio.run(select_model_endpoint(selected_model))
    if response is not None:
        st.write(f"Модель {selected_model} успешно выбрана")


# Инициализация состояния
if 'predictions_df' not in st.session_state:
    st.session_state.predictions_df = None
    st.session_state.response = None
    st.session_state.bet_size = 1000
    # Флаг для отслеживания первичного нажатия кнопки
    st.session_state.initial_prediction = False

# Кнопка для вызова endpoint
if st.button("Получить предсказания"):
    response = asyncio.run(get_predictions(st.session_state.bet_size))
    if response is not None:
        st.session_state.response = response
        st.session_state.predictions_df = pd.DataFrame(response["predictions"])
        st.session_state.initial_prediction = True  # Установка флага после первого нажатия


# Слайдер для выбора размера ставки только после первого нажатия кнопки
if st.session_state.initial_prediction:
    bet_size = st.slider("Размер ставки", min_value=0,
                                          max_value=10000,
                                          value=st.session_state.bet_size,
                                          step=100)

    # Если значение слайдера изменилось, предсказания обновляются
    if bet_size != st.session_state.bet_size:
        st.session_state.bet_size = bet_size  # Обновляем состояние слайдера
        response = asyncio.run(get_predictions(bet_size))
        if response is not None:
            st.session_state.response = response
            st.session_state.predictions_df = pd.DataFrame(response["predictions"])

# Если предсказания получены, отображаем результаты
if st.session_state.predictions_df is not None:
    st.write("Результаты предсказания:")
    st.dataframe(st.session_state.predictions_df)

    # Сheckbox для показа графика
    show_plot = st.checkbox(label='Показать график уверенности модели', value=False)

    # Если checkbox выбран, покажите график
    if show_plot:
        # Получиаем массив вероятностей
        y_proba = np.array(st.session_state.response["y_proba"])
        fig = plot_probabilities(y_proba)
        st.pyplot(fig)
