"""
Программа: Обучение модели
Версия: 1.0
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import f1_score
from lightgbm import LGBMClassifier
import optuna

from ..data.split_data import split_train_test
from ..train.metrics import save_metrics


def objective_lgb(trial,
                  x: pd.DataFrame,
                  y: pd.Series,
                  early_stopping_rounds: int,
                  n_folds: int = 5,
                  random_state: int = 10) -> np.array:
    """
    Целевая функция для поиска параметров
    :param trial: число итераций
    :param x: матрица объект-признак
    :param y: вектор целевой переменной
    :param early_stopping_rounds: число итераций, после которого нужно останавливать построение
    композиции при отсутствии улучшений
    :param n_folds: кол-во разбиений при кросс-валидации
    :param random_state: фиксатор эксперимента
    :return: среднее значение по метрике f1 на данной итерации
    """
    lgb_params = {
        "n_estimators": trial.suggest_categorical("n_estimators", [12199]),
        "learning_rate": trial.suggest_categorical("Learning_rate", [0.144]),
        "random_state": trial.suggest_categorical("random_state:", [random_state]),
        "is_unbalance": [True],
        "num_leaves": trial.suggest_int("num_leaves", 20, 1000, step=20),
        "max_bin": trial.suggest_int("max_bin", 200, 300),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.2, 0.99),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.2, 0.99),
    }

    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)

    cv_predicts = np.empty(n_folds)
    for idx, (train_idx, test_idx) in enumerate(cv.split(x, y)):
        x_train, x_test = x.iloc[train_idx], x.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        model = LGBMClassifier(**lgb_params)
        model.fit(x_train,
                  y_train,
                  eval_set=[(x_test, y_test)],
                  eval_metric=f1_metric,
                  early_stopping_rounds=early_stopping_rounds,
                  verbose=0)

        preds = model.predict(x_test)
        cv_predicts[idx] = f1_score(y_test, preds)

    return np.mean(cv_predicts)


def find_optimal_params(
        n_trials: int,
        x: pd.DataFrame,
        y: pd.Series,
        stratify: pd.Series,
        shuffle: bool,
        test_size: float,
        n_folds: int = 5,
        early_stopping_rounds: int = 100,
        random_state: int = 10) -> optuna.Study:
    """
    Нахождение оптимальных гиперпараметров модели
    :param early_stopping_rounds: число итераций, после которого нужно останавливать построение
    композиции при отсутствии улучшений
    :param n_trials: число итераций
    :param x: матрица объект признак
    :param y: целевая переменная
    :param stratify: имя переменной, для которой нужно сохранить баланс при разбиении
    :param shuffle: флаг, отвечающий за то, нужно ли перемешивать данные перед разбиением
    :param test_size: размер тестовой выборки
    :param n_folds: кол-во разбиений при кросс-валидации
    :param random_state: фиксатор эксперимента
    :return: объект класса optuna.Study, хранящий всю информацию об обучении
    """

    x_train, x_test, y_train, y_test = split_train_test(
        x=x,
        y=y,
        stratify=stratify,
        shuffle=shuffle,
        test_size=test_size,
        random_state=random_state,
    )

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        stratify=stratify,
        shuffle=shuffle,
        test_size=test_size,
        random_state=random_state
    )
    sampler = optuna.samplers.RandomSampler(seed=10)
    study = optuna.create_study(direction="maximize", study_name="LightGBM", sampler=sampler)
    func = lambda trial: objective_lgb(
        trial,
        x_train,
        y_train,
        n_folds=n_folds,
        random_state=random_state,
        early_stopping_rounds=early_stopping_rounds
    )

    study.optimize(func, n_trials=n_trials, show_progress_bar=True)

    return study


def f1_metric(labels: np.array, scores: np.array):
    """
    Реализация f1-метрики для LightGBM
    :param labels: метки классов
    :param scores: предсказанные вероятности принадлежности целевому классу
    :return:
    """
    pred = np.round(scores)
    return 'f1', f1_score(labels, pred), True


def train_model(
        x: pd.DataFrame,
        y: pd.Series,
        x_test: pd.DataFrame,
        y_test: pd.Series,
        stratify: pd.Series,
        shuffle: bool,
        test_size: int,
        early_stopping_rounds: int,
        random_state: int,
        study: optuna.Study,
        metric_path: str) -> LGBMClassifier:
    """
    Обучение модели на лучших параметрах
    :param x: матрица объект-признак (обучающая выборка)
    :param y: целевая переменная (обучающая выборка)
    :param x_test: матрица объект-признак (тестовая выборка)
    :param y_test: целевая переменная (тестовая выборка)
    :param stratify: переменная, для которой нужно сохранить баланс при разбиении
    :param shuffle: флаг, отвечающий за то, нужно ли перемешивать данные перед разбиением
    :param test_size: размер тестовой (валидационной) выборки
    :param early_stopping_rounds: число итераций, после которого нужно останавливать построение
    композиции при отсутствии улучшений
    :param random_state: фиксатор эксперимента
    :param study: объект класса optuna.Study, хранящий всю информацию об обучении
    :param metric_path: путь до пути с метриками
    :return: объект класса LGBMClassifier с подобранными параматрами
    """

    # Формирование валидационного множества для проверки целевой метрики после построения каждого нового дерева
    x_, x_val, y_, y_val = split_train_test(
        x,
        y,
        stratify=stratify,
        shuffle=shuffle,
        test_size=test_size,
        random_state=random_state
    )
    eval_set = [(x_val, y_val)]

    lgb_grid = LGBMClassifier(**study.best_params)
    lgb_grid.fit(x_,
                 y_,
                 eval_metric=f1_metric,
                 eval_set=eval_set,
                 verbose=2,
                 early_stopping_rounds=early_stopping_rounds)

    save_metrics(x=x_test, y=y_test, model=lgb_grid, metric_path=metric_path)
    return lgb_grid
