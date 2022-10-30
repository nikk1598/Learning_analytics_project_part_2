"""
Программа: Отрисовка слайдеров и кнопок для ввода данных
с дальнейшим получением предсказания на основании введенных значений
Версия: 1.0
"""

from io import BytesIO
import pandas as pd
import requests
import streamlit as st
from typing import Tuple, Dict


def evaluate(
        endpoint: str,
        files: Dict[str, Tuple[str, BytesIO, str]]):
    """
    Получение предсказаний из файла и вывод результата
    :param endpoint:
    :param files:
    :return:
    """
    button_ok = st.button("Predict")
    result = pd.DataFrame()

    if button_ok:
        with st.spinner('Модель делает предсказание...'):
            output = requests.post(
                endpoint,
                files=files,
                timeout=None)
            result["prediction"] = output.json()["prediction"]
            st.success('Успех!')
            st.write(result.head())
