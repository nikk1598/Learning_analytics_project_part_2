"""
Программа: Получение метрик
Версия: 1.0
"""

import pandas as pd
import json
import yaml
from sklearn.metrics import (
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    log_loss
)


def create_dict_metrics(y_test: pd.Series,
                        y_predict: pd.Series,
                        y_probability: pd.Series) -> dict:
    """
    Создание словаря с метриками для задачи бинарной классификации
    :param y_test: истинные метки
    :param y_predict: предсказанные метки
    :param y_probability: предсказанные вероятности отнесения к целевому классу
    :return: словарь с метриками
    """
    dict_metrics = {
        "roc_auc": round(roc_auc_score(y_test, y_probability[:, 1]), 3),
        "precision": round(precision_score(y_test, y_predict), 3),
        "recall": round(recall_score(y_test, y_predict), 3),
        "f1": round(f1_score(y_test, y_predict), 3),
        "logloss": round(log_loss(y_test, y_probability), 3)
    }
    return dict_metrics


def save_metrics(x: pd.DataFrame,
                 y: pd.Series,
                 model: object,
                 metric_path: str) -> None:
    """
    Сохранение словаря с метриками по заданному пути
    :param x: матрица объект-признак
    :param y: целевая переменная
    :param model: модель
    :param metric_path: путь для сохранения
    :return: None
    """
    result_metrics = create_dict_metrics(
        y_test=y,
        y_predict=model.predict(x),
        y_probability=model.predict_proba(x)
    )
    with open(metric_path, "w") as file:
        json.dump(result_metrics, file)


def load_metrics(config_path: str) -> dict:
    """
    Получение метрик из файла
    :param config_path: путь до конфигурационного файла
    :return: метрики
    """
    with open(config_path, encoding='utf-8') as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    with open(config['training']['metrics_path'], encoding='utf-8') as json_file:
        metrics = json.load(json_file)

    return metrics
