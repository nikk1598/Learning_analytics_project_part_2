## Описание проекта:

В мае-июне 2022 года на сайте ODS проводилось соревнование, в котором участникам надо было 
спрогнозировать академические задолженности у студентов с точностью до дисциплины. Ссылка на соревнование:
https://ods.ai/competitions/learning-analytics

Это вторая часть проекта, посвященная решению этой задачи. Её целью является создание приложения,
состоящего из backend и frontend-части, а также возможности запускать его с любой машины
при помощи программы Docker

На старте предполагается, что мы уже провели полное исследование, выбрали для решения 
модель LightGBM (несмотря на то, что Catboost и Stacking показали лучшие результаты по метрикам,
Stacking никто не использует в production из-за сложности работы с ним, а Catboost для нашей задачи
отрабатывает слишком уж долго, поэтому его использование будет не очень удобно для демонстрации),
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
        
Бинаризованные поля (каждый столбец представляет собой одно значение признака;
1 - значение соответствует объекту, 0 - не соответствует):
- MAIN_PLAN - учебный план
- PRED_ID - UID преподавателя
- DISC_DEP - факультет-организатор дисциплины
- CHOICE - выборность дисциплины
        
Целевой признак:
- DEBT (1 - есть задолженность, 0 - нет задолженности)
        
Таким образом, мы решаем задачу бинарной классификации, в которой объекты - это попытки, описанные
представленными выше признаками, а целевой признак говорит о факте наличия  или отсутствия долга.

## Описание модулей:

backend - backend-часть приложения:
- В src лежат основные модули, которые можно вызывать
для получения, разбиения и трансформации данных, для сохранения/загрузки метрик, а также для обучения модели Catboost
- В pipeline_training представлен полный процесс обучения, а в pipeline_evaluating - процесс получения предсказаний
с использованием описанных выше модулей
- В main.py методы pipeline_training и pipeline_evaluating используются для получения ответа на соответствующий запрос к
серверу при его поступлении, и запускается сам сервер

frontend - frontend-часть приложения:
- В main.py описан интерфейс страниц приложения - при выборе пользователем какой-нибудь страницы будет вызван соответствующий
метод. При этом, при нажатии на определённые кнопки внутри страниц, в этих методах будут  вызываться также методы из src, которые
нарисуют графики/отправят на сервер запрос (если речь идёт об обучении или предсказании), а также покажут ответ запроса
и, быть может, что-нибудь ещё

data - данные:
- df - датасет в формате parquet
- x_test  - тестовая выборка из этого датасета
- prediction - последнее сделанное предсказание 

models - модели:
- trained_model - последняя обученная модель
- study - объект study, хранящий в себе информацию о последнем обучении, а также подобранные гиперпараметры

report - метрики:
- metrics - метрики после последнего обучения

config - конфигурационный файл:
- params - файл со всеми константами, путями и т.д.

       
## Инструкция по запуску:
Чтобы запустить приложение, нужно:
1) установить docker
2) запустить docker
3) скачать папку с проектом
4) открыть папку с проектом в терминале 
5) прописать в терминале - docker compose up -d
6) в докере должен будет появиться контейнер, нужно навести на streamlit и нажать кнопку открыть
в браузере, либо, если кнопка недоступна, предварительно запустить всё приложение в верхнем правом углу

### Важное замечание
Чтобы при обучении подбирались гиперпараметры, нужно зайти в модуль pipeline_training и раскомментировать код,
который запускает поиск, а также закомментировать код, загружающий уже готовый объект study. По умолчанию
обучение будет идти, но уже с готовыми гиперпараметрами, чтобы долго не ждать, пока оно отработает
