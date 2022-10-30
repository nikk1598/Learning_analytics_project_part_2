"""
Программа: Полный пайплайн обучения модели
Версия: 1.0
"""

import yaml
import joblib
import os

from ..data.split_data import split_train_test
from ..data.get_data import get_dataset
from ..transform.transform import transform_types
from ..train.train import train_model


def pipeline_training(config_path: str) -> None:
    """
    Полный цикл получения данных предобработки и тренировки модели
    :param config_path: путь до конфигурационного файла
    :return: None
    """
    # Загрузка конфигурационного файла и получение параметров
    with open(config_path,  encoding='utf-8') as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    preprocessing_config = config['preprocessing']
    train_config = config['training']

    # Загрузка датасета
    df = get_dataset(dataset_path=preprocessing_config['df_path'])

    # Преобразование типов столбцов
    df = transform_types(
        data=df,
        change_type_columns=preprocessing_config['change_type_columns']
    )

    # Разбиение данных на обучающую и тестовую выборки
    x_train, x_test, y_train, y_test = split_train_test(
        x=df.drop(columns=[preprocessing_config['target_column']]),
        y=df[preprocessing_config['target_column']],
        stratify=df[preprocessing_config['target_column']],
        shuffle=preprocessing_config['shuffle'],
        test_size=preprocessing_config['test_size'],
        random_state=preprocessing_config['random_state']
        # x_test_path=preprocessing_config['x_test_path']
    )
    """
    # Поиск оптимальных гиперпараметров модели (занимает много времени, поэтому для демонстрации опустим эту часть)
    study = find_optimal_params(
        n_trials=train_config['n_trials'],
        x=x_train,
        y=y_train,
        early_stopping_rounds=train_config['early_stopping_rounds'],
        stratify=y_train,
        shuffle=train_config['shuffle'],
        test_size=train_config['test_size'],
        n_folds=train_config['n_folds'],
        random_state=train_config['random_state']
    )
    """

    # Загрузим объект study, хранящий в себе заранее подобранные гиперпараметры (см. закомментированный код выше)
    study = joblib.load(os.path.join(train_config['study_path']))

    # Обучение с лучшими параметрами
    clf = train_model(
        x=x_train,
        y=y_train,
        x_test=x_test,
        y_test=y_test,
        stratify=y_train,
        shuffle=train_config['shuffle'],
        test_size=train_config['test_size'],
        early_stopping_rounds=train_config['early_stopping_rounds'],
        random_state=train_config['random_state'],
        study=study,
        metric_path=train_config['metrics_path'],
    )

    # Сохранение результатов
    joblib.dump(clf, os.path.join(train_config['model_path']))
    joblib.dump(study, os.path.join(train_config['study_path']))
