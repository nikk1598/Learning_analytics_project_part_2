"""
Программа: Обучение модели на backend, отображение метрик и
графиков обучения на экране
Версия: 1.0
"""

import json
import os
import joblib
import requests
import streamlit as st
from optuna.visualization import plot_optimization_history
from lightgbm import plot_importance
from PIL import Image


def start_training(config: dict, endpoint: str) -> None:
    """
    Обучение модели с выводом результатов
    :param config: конфигурационный файл
    :param endpoint: endpoint
    :return: None
    """
    # Открываем файл с метриками, если он есть, и сохраняем их в переменную
    if os.path.exists(config['training']['metrics_path']):
        with open(config['training']['metrics_path']) as json_file:
            old_metrics = json.load(json_file)
    # Если файла нет, то зануляем все метрики
    else:
        old_metircs = {'roc_auc': 0,
                       'precision': 0,
                       'recall': 0,
                       'f1': 0,
                       'logloss': 0}
    ''' 
    Показываем колесо и отправляем запрос (в main сюда будет 
    передан адрес, по которому пользователь может обучить модель,
    соответственно медод из модуля main бэкэнда начнёт выполнять соответствующий код)
    '''
    with st.spinner('Модель подбирает параметры, процесс может занимать до 5 минут...'):
        '''
        В output запишется то, что возвращает сервер в ответ на запрос
        (как мы помним, для запроса на обучение это словарь с метриками плюс
        в old_metrics у нас уже сохранены предыдущие метрики, либо нули)
        '''
        output = requests.post(endpoint, timeout=None)
    st.success('Успех!')

    '''
    Представляем наши метрики в формате специальной таблицы streamlit,
    для каждого столбца передаём название метрики, новую подсчитанную метрику,
    а также разницу между старой и новой метрикой
    '''
    new_metrics = output.json()
    roc_auc, precision, recall, f1, logloss = st.columns(5)
    roc_auc.metric(
        'ROC_AUC',
        new_metrics['roc_auc'],
        f"{new_metrics['roc_auc'] - old_metrics['roc_auc']:.3f}"
    )
    precision.metric(
        'Precision',
        new_metrics['precision'],
        f"{new_metrics['precision'] - old_metrics['precision']:.3f}"
    )
    recall.metric(
        'Recall',
        new_metrics['recall'],
        f"{new_metrics['recall'] - old_metrics['recall']:.3f}"
    )
    f1.metric(
        'F1-score',
        new_metrics['f1'],
        f"{new_metrics['f1'] - old_metrics['f1']:.3f}"
    )
    logloss.metric(
        'Logloss',
        new_metrics['logloss'],
        f"{new_metrics['logloss'] - old_metrics['logloss']:.3f}"
    )

    # Показываем графики с результатами обучения
    study = joblib.load(os.path.join(config['training']['study_path']))
    clf = joblib.load(os.path.join(config['training']['model_path']))

    fig_history = plot_optimization_history(study)
    st.plotly_chart(fig_history, use_container_width=True)

    feat_imp = plot_importance(clf, max_num_features=10, figsize=(20, 10))
    feat_imp.figure.savefig(config['report']['feat_imp_path'])

    image = Image.open(config['report']['feat_imp_path'])
    st.image(image, caption='Features')
