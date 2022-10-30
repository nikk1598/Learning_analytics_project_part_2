"""
Программа: Полный пайплайн получения предсказаний на основе обученной модели
Версия: 1.0
"""

import yaml
import joblib
import os
from typing import BinaryIO
import json

from ..data.get_data import get_dataset
from ..transform.transform import transform_types


def pipeline_evaluate(
    config_path: str,
    data_path: BinaryIO,
) -> list:
    """
    Предобработка входных данных и получение предсказаний
    :param config_path: путь до конфигурационного файла
    :param data_path: путь до датасета
    :return: предсказания
    """
    # Загрузка конфигурационного файла
    with open(config_path, encoding='utf-8') as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    preprocessing_config = config['preprocessing']
    train_config = config['training']

    # Получение датасета по заданному пути
    df = get_dataset(dataset_path=data_path)

    # Преобразование типов столбцов
    df = transform_types(
        data=df,
        change_type_columns=preprocessing_config['change_type_columns']
    )

    # Загрузка обученной модели
    model = joblib.load(os.path.join(train_config['model_path']))

    # Получение предсказаний
    prediction = model.predict(df).tolist()

    with open(config['prediction']['prediction_path'], "w") as file:
        json.dump(prediction, file)

    return prediction
