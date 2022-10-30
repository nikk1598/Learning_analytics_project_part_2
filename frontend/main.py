"""
Программа: Frontend часть проекта
Версия: 1.0
"""

import streamlit as st
import yaml
import os

from src.plotting import charts
from src.data import get_data
from src.train import train
from src.evaluate.evaluate import evaluate

st.set_option('depr''ecation.showPyplotGlobalUse', False)

CONFIG_PATH = '../config/params_.yml'


def main_page():
    """
    Страница с описанием проекта
    :return:
    """
    st.image(
        "https://media.giphy.com/media/uyZPevMqDJ3r76phm1/giphy.gif",
        width=550
    )

    st.markdown("# Описание проекта")
    st.title("MLOps project: Learning Analytics Competition")
    st.write(
        """
       В мае-июне 2022 года на сайте ODS проводилось соревнование, в котором участникам надо было спрогнозировать
       академические задолженности у студентов с точностью до дисциплины. Ссылка на соревнование:
       https://ods.ai/competitions/learning-analytics

       Это вторая часть проекта, посвященная решению этой задачи. Её целью является создание приложения, состоящего
       из сервера и frontend-части, а также возможности запускать его с любой машины при помощи программы Docker

       На старте предполагается, что мы уже провели полное исследование, выбрали для решения модель LightGBM,
       а также сформировали датасет, состоящий из следующих столбцов:

       - SEMESTER - семестр получения оценки
       - DISC_ID - UID дисциплины
       - TYPE_NAME - форма отчётности
       - GENDER - пол студента
       - CITIZENSHIP - гражданство студента
       - EXAM_TYPE - форма зачисления студента (ЕГЭ, олимпиада, ВИ - вступительные испытания)
       - EXAM_SUBJECT_1 - первый экзамен ЕГЭ
       - EXAM_SUBJECT_2 - второй экзамен ЕГЭ
       - EXAM_SUBJECT_3 - третий экзамен ЕГЭ
       - ADMITTED_EXAM_1 - баллы за 1 экзамен ЕГЭ
       - ADMITTED_EXAM_2 - баллы за 2 экзамен ЕГЭ
       - ADMITTED_EXAM_3 - баллы за 3 экзамен ЕГЭ
       - ADMITTED_SUBJECT_PRIZE_LEVEL - уровень олимпиады (если есть)
       - REGION_ID - номер региона студента
       
       Бинаризованные поля (каждый столбец представляет собой одно значение признака; 1 - значение соответствует
       объекту, 0 - не соответствует):

       - MAIN_PLAN - учебный план
       - PRED_ID - UID преподавателя
       - DISC_DEP - факультет-организатор дисциплины
       - CHOICE - выборность дисциплины
       
       Целевой признак:

       - DEBT (1 - есть задолженность, 0 - нет задолженности)
       Таким образом, мы решаем задачу бинарной классификации, в которой объекты - это попытки, описанные
        представленными выше признаками, а целевой признак говорит о факте наличия или отсутствия долга
       """
    )


def exploratory():
    """
    Exploratory data analysis
    """
    st.markdown('# Exploratory data analysis')

    with open(CONFIG_PATH, encoding='utf8') as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    data = get_data.get_data(config['preprocessing']['df_path'])
    st.write(data.head())

    target_values_frequency = st.sidebar.checkbox('Распределение target')
    target_gender = st.sidebar.checkbox('Распределение target в разрезе пола')
    target_exam = st.sidebar.checkbox('Распределение target в разрезе типа экзамена')
    target_type_name = st.sidebar.checkbox('Распределение target в разрезе типа отчётности')
    mean_score_target = st.sidebar.checkbox('Распределение среднего балла ЕГЭ/вступительных экзаменов '
                                            'в разрезе target (график плотности)')
    mean_score_target_boxlot = st.sidebar.checkbox('Распределение среднего балла ЕГЭ/вступительных экзаменов'
                                                   ' в разрезе target (график boxplot)')

    if target_values_frequency:
        st.pyplot(
            charts.barplot(
                df=data,
                col=config['preprocessing']['target_column'],
                title='Распределение классов в разрезе target')
        )

    if target_gender:
        st.pyplot(
            charts.barplot_group(
                df=data,
                col=config['preprocessing']['target_column'],
                col_group=config['preprocessing']['gender_column'],
                values_in_col_group=config['preprocessing']['values_in_gender_column'],
                title='Распределение target в разрезе пола')
        )

    if target_exam:
        st.pyplot(
            charts.barplot_group(
                df=data,
                col=config['preprocessing']['target_column'],
                col_group=config['preprocessing']['exam_type_column'],
                values_in_col_group=config['preprocessing']['values_in_exam_type_column'],
                title='Распределение target в разрезе типа экзамена')
        )

    if target_type_name:
        st.pyplot(
            charts.barplot_group(df=data,
                                 col=config['preprocessing']['target_column'],
                                 col_group=config['preprocessing']['type_name_column'],
                                 values_in_col_group=config['preprocessing']['values_in_type_name_column'],
                                 title='Распределение target в разрезе типа отчётности')
        )

    if mean_score_target:
        data = charts.create_mean_column(
            data=data,
            cols=config['preprocessing']['addmited_exam_columns']
        )

        st.pyplot(
            charts.displots_of_statistic(
                df=data,
                col='mean',
                col_group=config['preprocessing']['target_column'],
                values_in_col_group=config['preprocessing']['values_in_target_column'],
                title='Распределение значений mean_score\n')
        )
        del data['mean']

    if mean_score_target_boxlot:
        data = charts.create_mean_column(
            data=data,
            cols=config['preprocessing']['addmited_exam_columns']
        )

        st.pyplot(
            charts.boxplot(
                df=data,
                x=config['preprocessing']['target_column'],
                y='mean',
                title='Распределение значений mean_score\n')
        )
        del data['mean']


def training():
    """
    Обучение модели
    """
    with open(CONFIG_PATH, encoding='utf8') as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    st.markdown("# Training model LightGBM")
    endpoint = config['endpoints']['train']

    if st.button("Start training"):
        train.start_training(
            config=config,
            endpoint=endpoint
        )


def prediction():
    """
    Получение предсказаний модели из файла с данными
    """
    st.markdown("# Prediction")

    with open(CONFIG_PATH, encoding='utf8') as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    endpoint = config['endpoints']['prediction']

    upload_file = st.file_uploader(
        "", type=['parquet.gzip'], accept_multiple_files=False
    )

    if upload_file:
        files = get_data.load_data(data=upload_file, type_data='Test')
        if os.path.exists(config['training']['model_path']):
            evaluate(endpoint=endpoint, files=files)


if __name__ == "__main__":
    page_names = {
        'Описание проекта': main_page,
        'Exploratory data analysis': exploratory,
        'Training model': training,
        'Prediction': prediction,
    }
    selected_page = st.sidebar.selectbox('Выберите пункт', page_names.keys())
    page_names[selected_page]()
