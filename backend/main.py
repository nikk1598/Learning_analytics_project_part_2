"""
Программа: Модель для прогнозирования наличия задолженности
Версия: 1.0
"""

import warnings
import optuna
import uvicorn
from fastapi import FastAPI
from fastapi import File
from fastapi import UploadFile

from src.pipelines.pipeline_training import pipeline_training
from src.pipelines.pipeline_evaluating import pipeline_evaluate
from src.train.metrics import load_metrics

warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

app = FastAPI()
CONFIG_PATH = '../config/params_.yml'


@app.post('/train')
def training():
    """
    Обучение модели
    """
    pipeline_training(config_path=CONFIG_PATH)
    metrics = load_metrics(config_path=CONFIG_PATH)
    return metrics


@app.post('/predict')
def prediction(file: UploadFile = File(...)):
    """
    Предсказание модели по данным из файла
    """
    result = pipeline_evaluate(
        config_path=CONFIG_PATH,
        data_path=file.file,
    )

    assert isinstance(result, list), "Результат не соответствует типу list"
    return {"prediction": result}


if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=80)
