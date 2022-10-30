"""
Программа: Получение данных из файла
Версия: 1.0
"""

from typing import Text, Union, BinaryIO
import pandas as pd


def get_dataset(dataset_path: Union[Text, BinaryIO]) -> pd.DataFrame:
    """
    Получение данных в формате parquet по заданному пути
    :param dataset_path: путь до датасета
    :return: датасет
    """
    return pd.read_parquet(dataset_path)
