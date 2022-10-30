"""
Программа: Предобработка данных
Версия: 1.0
"""

import pandas as pd


def transform_types(data: pd.DataFrame,
                    change_type_columns: dict) -> pd.DataFrame:
    """
    Преобразование набора признаков в заданный тип данных
    :param data: датасет
    :param change_type_columns: словарь с признаками и типами данных
    :return: преобразованный датасет
    """
    return data.astype(change_type_columns, errors="raise")
