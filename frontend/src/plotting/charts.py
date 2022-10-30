"""
Программа: Отрисовка графиков
Версия: 1.0
"""

import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib import rcParams
import pandas as pd


def plot_text(ax):
    """
    Добавление подписи процентов на график barplot
    :param ax: ось
    :return: None
    """
    for p in ax.patches:
        percentage = '{:.1f}%'.format(p.get_height())
        ax.annotate(
            percentage,  # текст
            # координата xy
            (p.get_x() + p.get_width() / 2., p.get_height()),
            # центрирование
            ha='center',
            va='center',
            xytext=(0, 10),
            # точка смещения относительно координаты
            textcoords='offset points',
            fontsize=14)


def barplot(df: pd.DataFrame,
            col: str,
            title: str) -> None:
    """
    Построение графика распределения признака в виде столбчатой диаграммы
    :param df: датасет
    :param col: столбец, для которого которого хотим смотреть распределение
    :param title: заголовок
    :return: None
    """
    rcParams['figure.figsize'] = 10, 8
    sns.color_palette("YlOrBr", as_cmap=True)

    # Датафрейм частот значений
    norm_target = pd.DataFrame(df[col].value_counts(normalize=True).mul(100) \
                               .rename('percent')).reset_index()

    ax = sns.barplot(x='index', y='percent', data=norm_target, palette="flare")
    plt.title(title)
    plot_text(ax)


def barplot_group(df: pd.DataFrame,
                  col: str,
                  col_group: str,
                  values_in_col_group: list,
                  title: str) -> None:
    """
    Построение графика распределения признака в виде ступенчатой диаграммы
    в разрезе другого бинарного признака
    :param df: датасет
    :param col: столбец, для которого хотим смотреть распределение
    :param col_group: столбец, в разрезе которого хотим смотреть распределение
    :param values_in_col_group: список значений col_group
    :param title: заголовок
    :return: None
    """
    rcParams['figure.figsize'] = 10, 8
    sns.color_palette("YlOrBr", as_cmap=True)

    # Датафрейм частот для каждого значения col_group
    dataframes_of_frequency = []

    for x in values_in_col_group:
        freq = df[df[col_group] == x][col] \
            .value_counts(normalize=True).rename('percent').reset_index()
        freq[col_group] = x
        dataframes_of_frequency.append(freq)

    # Общий датафрейм частот
    target_values = pd.concat(dataframes_of_frequency)
    target_values.rename(columns={"index": "target"}, inplace=True)
    target_values['percent'] = target_values['percent'] * 100

    g = sns.catplot(x=col_group,
                    y='percent',
                    hue='target',
                    data=target_values,
                    kind='bar',
                    height=8,
                    palette=sns.color_palette(["indianred", "purple"]))
    plt.title(title)

    plot_text(g.ax)


def displots_of_statistic(df: pd.DataFrame,
                          col: str,
                          col_group: str,
                          values_in_col_group: list,
                          title: str) -> None:
    """
    Построение графиков распределений значений одного столбца в разрезе значений
     другого столбца
    :param df: датасет
    :param col: столбец, по которому хотим смотреть распределение
    :param col_group: столбец, в разрезе которого хотим смотреть распределение
    :param values_in_col_group: список значений col_group
    :param title: заголовок
    :return: None
    """

    ''' Словарь, в котором значения являются подстолбцами нужного нам столбца, соответствующие
    той или иной градации признака col_group. Ключи, по сути, являются идентификатором
    этой градации
    '''
    data_for_displot = {}
    for x in values_in_col_group:
        data_for_displot[col_group + ' ' + str(x)] = df[df[col_group] == x][col]

    sns.displot(
        data=data_for_displot,
        kind="kde",
        common_norm=False)

    plt.title(title)


def boxplot(df: pd.DataFrame,
            x: str,
            y: str,
            title: str) -> None:
    """
    Построение графиков boxplot по столбцу в разрезе значений другого столбца
    :param df: датасет,
    :param x: столбец, в разрезе которого хотим строить график
    :param y: столбец, по которому хотим строить график
    :param title: заголовок
    :return: None
    """
    sns.boxplot(x=x, y=y, data=df)
    plt.title(title, fontsize=20)
    plt.ylabel(y, fontsize=14)
    plt.xlabel(x, fontsize=14)


def create_mean_column(data: pd.DataFrame,
                       cols: list) -> pd.DataFrame:
    """
    Добавление столбца средних по некоторому подмножеству признаков (столбцов)
    :param data: датасет
    :param cols: список столбцов, значения которых участвуют в вычислении среднего
    """
    # Столбец суммы по подмножеству признаков
    sum_ = 0
    for x in cols:
        sum_ += data[x]

    # Столбец средних
    data['mean'] = sum_ / len(cols)
    return data
