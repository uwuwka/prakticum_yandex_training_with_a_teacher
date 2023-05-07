#!/usr/bin/env python
# coding: utf-8

# <div style="background: #C0C0C0; padding: 5px; border: 1px solid black; border-radius: 5px;">
#     <font color='black'><u><b>КОММЕНТАРИЙ РЕВЬЮЕРА</b></u></font><br />
#     <br />
#     Привет, Ратимир! Меня зовут Кирилл Киселев.<br />
#         На этом проекте я буду ревьюером. Предлагаю общаться на "ты". Если так будет неудобно, пожалуйста, дай знать.<br />
#         <br />
#         Моя цель - помочь тебе освоиться в новой профессии и научиться применять полученные знания максимально эффективно. Не стесняйся спрашивать, если что-то непонятно. Вместе проще разобраться с любыми вопросами.<br />
#         <br />
# В проекте тебе встретятся мои комментарии к коду и выводам. Пожалуйста, не удаляй их, так будет удобнее вести общение.<br />
#         Ниже приведены примеры оформления моих комментариев:<br />
# </div>
# <div class="alert alert-success" style="border-color: green; border-radius: 5px">
#     <p><u><b>✅ КОММЕНТАРИЙ РЕВЬЮЕРА</b></u></p>
#     <p>Таким образом оформляется комментарий, означающий что пункт выполнен без ошибок.</p>
# </div>
# <div class="alert alert-warning" style="border-color: orange; border-radius: 5px">
#     <p><u><b>⚠️ КОММЕНТАРИЙ РЕВЬЮЕРА</b></u></p>
#     <p>Таким образом оформляется комментарий, означающий что в пункте есть некритичные недочеты.</p>
# </div>
# <div class="alert alert-danger" style="border-color: #8B0000; border-radius: 5px">
#     <p><u><b>❌ КОММЕНТАРИЙ РЕВЬЮЕРА</b></u></p>
#     <p>Таким образом оформляется комментарий, означающий что в пункте есть критичные недочеты, требующие исправления.</p>
# </div>
# <div class="alert alert-info" style="border-color: #0080FF; border-radius: 5px">
#     <p><u><b>КОММЕНТАРИЙ СТУДЕНТА</b></u></p>
#     <p>Буду очень признателен, если ты тоже оформишь свои комментарии и вопросы ко мне с прменением цветовой разметки. Например, так.</p>     
# </div>

# <h1>Содержание<span class="tocSkip"></span></h1>
# <div class="toc"><ul class="toc-item"><li><span><a href="#Подготовка-данных" data-toc-modified-id="Подготовка-данных-1"><span class="toc-item-num">1&nbsp;&nbsp;</span>Подготовка данных</a></span></li><li><span><a href="#Исследование-задачи" data-toc-modified-id="Исследование-задачи-2"><span class="toc-item-num">2&nbsp;&nbsp;</span>Исследование задачи</a></span></li><li><span><a href="#Борьба-с-дисбалансом" data-toc-modified-id="Борьба-с-дисбалансом-3"><span class="toc-item-num">3&nbsp;&nbsp;</span>Борьба с дисбалансом</a></span></li><li><span><a href="#Тестирование-модели" data-toc-modified-id="Тестирование-модели-4"><span class="toc-item-num">4&nbsp;&nbsp;</span>Тестирование модели</a></span></li><li><span><a href="#Вывод" data-toc-modified-id="Вывод-5"><span class="toc-item-num">5&nbsp;&nbsp;</span>Вывод</a></span></li><li><span><a href="#Чек-лист-готовности-проекта" data-toc-modified-id="Чек-лист-готовности-проекта-6"><span class="toc-item-num">6&nbsp;&nbsp;</span>Чек-лист готовности проекта</a></span></li><li><span><a href="#Финальные-комментарии-ревьюера" data-toc-modified-id="Финальные-комментарии-ревьюера-7"><span class="toc-item-num">7&nbsp;&nbsp;</span>Финальные комментарии ревьюера</a></span></li></ul></div>

# # Отток клиентов

# Из «Бета-Банка» стали уходить клиенты. Каждый месяц. Немного, но заметно. Банковские маркетологи посчитали: сохранять текущих клиентов дешевле, чем привлекать новых.
# 
# Нужно спрогнозировать, уйдёт клиент из банка в ближайшее время или нет. Вам предоставлены исторические данные о поведении клиентов и расторжении договоров с банком. 
# 
# Постройте модель с предельно большим значением *F1*-меры. Чтобы сдать проект успешно, нужно довести метрику до 0.59. Проверьте *F1*-меру на тестовой выборке самостоятельно.
# 
# Дополнительно измеряйте *AUC-ROC*, сравнивайте её значение с *F1*-мерой.
# 
# Источник данных: [https://www.kaggle.com/barelydedicated/bank-customer-churn-modeling](https://www.kaggle.com/barelydedicated/bank-customer-churn-modeling)

# <div class="alert alert-success" style="border-color: green; border-radius: 5px">
#     <p><u><b>✅ КОММЕНТАРИЙ РЕВЬЮЕРА v.1</b></u></p>
#     <p>Содержание и формулировка задачи на месте. Это важный структурный компанент проекта. Отлично!</p>
# </div>

# ## Подготовка данных

# In[1]:


from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, roc_auc_score, roc_curve, confusion_matrix
from sklearn.preprocessing import OneHotEncoder

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.dummy import DummyClassifier


import warnings
warnings.filterwarnings("ignore")


# <div class="alert alert-success" style="border-color: green; border-radius: 5px">
#     <p><u><b>✅ КОММЕНТАРИЙ РЕВЬЮЕРА v.1</b></u></p>
#     <p>Хорошо, что импортируешь нужные библиотеки на первом этапе. Так те, кто будет запускать твой проект в будущем сразу будут знать, какие библиотеки необходимо установить.</p>
# </div>

# сохраняем данные в переменную и смотрим общую информацию

# In[2]:


df = pd.read_csv('/datasets/Churn.csv')
df.head()


# In[3]:


df.info()


# <div class="alert alert-success" style="border-color: green; border-radius: 5px">
#     <p><u><b>✅ КОММЕНТАРИЙ РЕВЬЮЕРА v.1</b></u></p>
#     <p>Данные загружены, отлично.</p>
#     <p>Хорошо, что применяешь метод <code>.head()</code> таким образом мы можем визуально оценить структуру датасета.</p>
#     <p>Метод <code>.info()</code> обязателен к применению в таких задачах. И он на месте.</p>
#     <p>Дополнительно можешь применить метод <code>.describe()</code>, в совокупности с <code>.info()</code> и <code>.head()</code> можно получить первые инсайты из данных.</p>
# </div>

# Проверим на дубликаты

# In[4]:


df.duplicated().sum()


# <div class="alert alert-success" style="border-color: green; border-radius: 5px">
#     <p><u><b>✅ КОММЕНТАРИЙ РЕВЬЮЕРА v.1</b></u></p>
#     <p>Хорошо, что проверила на дубликаты, это важно!</p>
# </div>

# заполняем пропуски минимальными значениями

# In[5]:


df['Tenure'] = df['Tenure'].fillna(0)


# <div class="alert alert-success" style="border-color: green; border-radius: 5px">
#     <p><u><b>✅ КОММЕНТАРИЙ РЕВЬЮЕРА v.1</b></u></p>
#     <p>Ты выбрала один из лучших вариантов для заполнения пропусков в данном случае. Также можно было заполнить просто медианным значением.</p>
#     <p>К заполнению пропусков можно вернуться после получения результатов работы моделей и попробовать другие варианты.</p>
#     <p>А в реальной жизни с этим вопросом можно обратиться к коллегам, чтобы уточнить причины возникновения пропущенных значений.</p>
# </div>

# ## Исследование задачи

# Удалим столбцы-идентификаторы, не представляющие ценностия для алгоритма.
# создадим дополнительный датафрем, в котором будут хранится данные для обучения модели

# In[6]:


drop_columns = ['RowNumber','CustomerId', 'Surname']
df_ml = df.drop(drop_columns, axis=1)


# <div class="alert alert-success" style="border-color: green; border-radius: 5px">
#     <p><u><b>✅ КОММЕНТАРИЙ РЕВЬЮЕРА v.1</b></u></p>
#     <p>Столбцы для удаления выбраны верно. Молодец!</p>
# </div>

# In[7]:


df_ml.shape


# Данные подготовим методом OHE, что позволит нам использовать разные модели и не словить дамми ловушку

# In[8]:


df_ml = pd.get_dummies(df_ml, drop_first=True)


# <div class="alert alert-warning" style="border-color: orange; border-radius: 5px">
#     <p><u><b>⚠️ КОММЕНТАРИЙ РЕВЬЮЕРА v.1</b></u></p>
#     <p>Тут стоит обсудить кодирование категориальных признаков.</p>
#     <p>Ты безусловно верно применил <code>pd.get_dummies</code>, однако, как показывает практика, этот метод имеет ряд существенных недостатков, поскольку кодирует весь датасет целиком. То есть и трейн и валидацию и тест. Но чаще всего тестовый набор у нас вроде как «слепой» в котором могут быть категории, которых не было изначально в данных на которых обучалась модель, поэтому я рекомендую использовать <code>OneHotEncoder</code> из библиотеки <code>sklearn</code> вместо <code>get_dummies</code> из <code>pandas</code>. В проекте «Обучение с учителем» таких проблем не возникает. Однако в последующих проектах кодирование до деления может привести к ошибкам.</p>
#     <p>То есть верный алгоритм преобразования следующий:</p>
#     <p>1. Выполняешь разделение данных на выборки.</p>
#     <p>2. Обучаешь <code>OneHotEncoder</code> на тренировочной выборке.</p>
#     <p>3. Трансформируешь все выборки.</p>
#     <p>Оставлю пару полезных ссылок по этой теме.</p>
#     
# https://stackoverflow.com/questions/55525195/do-i-have-to-do-one-hot-encoding-separately-for-train-and-test-dataset
# https://albertum.medium.com/preprocessing-onehotencoder-vs-pandas-get-dummies-3de1f3d77dcc
# </div>

# In[9]:


df_ml.shape


# Разделим на признаки и целевой признак

# In[10]:


features = df_ml.drop('Exited', axis=1)
target = df_ml['Exited']


# In[11]:


features_train, features_test, target_train, target_test = train_test_split(features,
                                                    target,
                                                    train_size=0.6,
                                                    random_state=12345)
features_valid, features_test, target_valid, target_test = train_test_split(features_test,
                                                    target_test,
                                                    train_size=0.5,
                                                    random_state=12345)


# посмотрим исходные валидационные данные, чтобы сравнить их с от маштабированными в дальнейшем

# In[12]:


features_valid.head()


# In[13]:


features_valid.shape


# посмотрим исходные тренировочные данные, чтобы сравнить их с от маштабированными в дальнейшем

# In[14]:


features_train.head()


# In[15]:


features_train.shape


# посмотрим исходные тестовые данные, чтобы сравнить их с от маштабированными в дальнейшем

# In[16]:


features_train.head()


# In[17]:


features_test.shape


# In[18]:


enc = OneHotEncoder(handle_unknown = 'ignore')
enc.fit(features_train)


# In[19]:


enc.transform(features_train)
enc.transform(features_test)
enc.transform(features_valid)
enc.transform(features_test)


# <div class="alert alert-danger" style="border-color: #8B0000; border-radius: 5px">
#     <p><u><b>❌ КОММЕНТАРИЙ РЕВЬЮЕРА v.1</b></u></p>
#     <p>Алгоритмически разделение выполнено верно. Напомню, что на этом шаге нам нужно получить три выборки обучающую, валидационную и тестовую в соотношении 3/1/1. Поправь пожалуйста!</p>
# </div>

# <div class="alert alert-danger" style="border-color: #8B0000; border-radius: 5px">
#     <p><u><b>❌ КОММЕНТАРИЙ РЕВЬЮЕРА v.2</b></u></p>
#     <p>При разделении датасета в соотношении 3/1/1 у тебя должны получиться выборки по 6000/2000/2000 строк соответственно. У тебя получились выборки размером 6000/800/3200. Это неверно.</p>
# </div>

# <div class="alert alert-success" style="border-color: green; border-radius: 5px">
#     <p><u><b>✅ КОММЕНТАРИЙ РЕВЬЮЕРА v.3</b></u></p>
#     <p>Тепепь ок.</p>
# </div>

# Для масштабирования методом scaler зафиксируем численные признаки

# In[20]:


numeric = ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'EstimatedSalary']


# In[21]:


scaler = StandardScaler()
scaler.fit(features_train[numeric])


# Масштабируем численные признаки обучающей выборки

# In[22]:


features_train[numeric] = scaler.transform(features_train[numeric])
features_train.head()


# Масштабируем численные признаки валидационной выборки 

# In[23]:


features_valid[numeric] = scaler.transform(features_valid[numeric])
features_valid.head()


# Масштабируем численные признаки тестовой выборки 

# In[24]:


features_test[numeric] = scaler.transform(features_test[numeric])
features_test.head()


# <div class="alert alert-success" style="border-color: green; border-radius: 5px">
#     <p><u><b>✅ КОММЕНТАРИЙ РЕВЬЮЕРА v.1</b></u></p>
#     <p>Отлично, выборки подготовлены и отмасштабированы. Все верно.</p>
# </div>

# строим диаграмму, которая наглядно покажет дисбаланс данных на тренировочной выборке

# In[25]:


df_ml['Exited'].plot(kind ='hist', bins=2, figsize=(5,5))


# строим диаграмму на тестовой выборке, чтобы убедиться в том, что дисбаланс есть и это не простая ошибка в тренировочной выборке.

# Как мы выяснили в нашей выборке отрицательны ответов ≈80% , положитительных ≈ 20%. С уверенностью можем сказать что дисбаланс присутствует и он существенный.

# <div class="alert alert-danger" style="border-color: #8B0000; border-radius: 5px">
#     <p><u><b>❌ КОММЕНТАРИЙ РЕВЬЮЕРА v.1</b></u></p>
#     <p>В этом пункте дополнительно необходимо исслеовать баланс классов.</p>
#     <p>Нам нужно узнать как соотносятся классы в целевом признаке.</p>
# </div>

# <div class="alert alert-warning" style="border-color: orange; border-radius: 5px">
#     <p><u><b>⚠️ КОММЕНТАРИЙ РЕВЬЮЕРА v.2</b></u></p>
#     <p>Фактически исследование дисбаланса есть.</p>
#     <p>Однако его стоило выполнить на всем датасете, а не на отдельных выборках. Гистограмму можно было выполнить крупнее, чтобы она была более наглядной.</p>
# </div>

# <div class="alert alert-success" style="border-color: green; border-radius: 5px">
#     <p><u><b>✅ КОММЕНТАРИЙ РЕВЬЮЕРА v.3</b></u></p>
#     <p>Так значительно лучше))</p>
# </div>

# напишем функцию для изучия полноты, точности и F1-меры

# In[26]:


def rec_prec_f1(target_valid, prediction):
    print("Полнота" , recall_score(target_valid, prediction))
    print("Точность", precision_score(target_valid, prediction))
    print("F1-мера", f1_score(target_valid, prediction))


# обучаем модели

# In[27]:


best_RF = None
best_accuracy_RF = 0
best_f1_RF=0
best_est_RF = 0
best_depth_RF = 0
for est in tqdm(range(10,200,10)):
    for depth in range(2,25):
        RF = RandomForestClassifier(random_state = 12345,n_estimators = est, max_depth = depth)
        RF.fit(features_train,target_train)
        prediction_valid_RF = RF.predict(features_valid)
        accuracy_RF = accuracy_score(prediction_valid_RF, target_valid)
        f1_RF = f1_score(prediction_valid_RF, target_valid)
        if best_f1_RF < f1_RF:
            best_f1_RF = f1_RF
            best_RF = RF
            best_depth_RF = depth
            best_est_RF = est
            best_accuracy_RF = accuracy_RF

RF_probabilities_one_valid = best_RF.predict_proba(features_valid)[:, 1]

print(rec_prec_f1(target_valid,prediction_valid_RF))
print('AUC-ROC',roc_auc_score(target_valid, RF_probabilities_one_valid))
print()
print('лучшая модель с параметрами:', 'глубина-',best_depth_RF,'','количество ветвей-',best_est_RF)


# In[28]:


best_DT = None
best_accuracy_DT = 0
best_f1_DT = 0
best_depth_DT = 0
for depth in tqdm(range(2,52)):
    DT = DecisionTreeClassifier(random_state = 12345, max_depth = depth)
    DT.fit(features_train,target_train)
    prediction_valid_DT = DT.predict(features_valid)
    accuracy_DT = accuracy_score(prediction_valid_DT, target_valid)
    f1_DT = f1_score(prediction_valid_DT, target_valid)
    if best_f1_DT < f1_DT:
        best_f1_DT = f1_DT
        best_DT = DT
        best_accuracy_DT = accuracy_DT
        best_depth_DT = depth
        
DT_probabilities_one_valid = best_DT.predict_proba(features_valid)[:, 1]

print(rec_prec_f1(target_valid,prediction_valid_DT))
print('AUC-ROC',roc_auc_score(target_valid, DT_probabilities_one_valid))
print()
print('лучшая модель с параметрами:', 'глубина-',best_depth_DT,)


# In[29]:


best_LR = None
best_f1_LR=0
best_accuracy_LR = 0
for iter_ in tqdm([100, 200, 500, 1000, 5000]):
    LR = LogisticRegression(random_state = 12345,max_iter=iter_,tol=1e-5,solver = 'lbfgs' )
    LR.fit(features_train,target_train)
    prediction_valid_LR = LR.predict(features_valid)
    accuracy_LR = accuracy_score(prediction_valid_LR, target_valid)
    f1_LR = f1_score(prediction_valid_LR, target_valid)
    if best_f1_LR < f1_LR:
        best_LR = LR
        best_accuracy_LR = accuracy_LR
            
LR_probabilities_one_valid = best_LR.predict_proba(features_valid)[:, 1]            
            
print(rec_prec_f1(target_valid,prediction_valid_LR))
print('AUC-ROC',roc_auc_score(target_valid, LR_probabilities_one_valid))


# <div class="alert alert-danger" style="border-color: #8B0000; border-radius: 5px">
#     <p><u><b>❌ КОММЕНТАРИЙ РЕВЬЮЕРА v.1</b></u></p>
#     <p>Давай при оптимизации моделей анализировать не accuracy а f1-меру. У нас же дисбаланс классов.</p>
#     <p>После подбора оптимальных гиперпараметров необходимо вывести их на печать.</p>
#     <p>Здесь нужно дополнительно рассчитать метрику AUC-ROC для всех моделей.</p>
# </div>

# <div class="alert alert-danger" style="border-color: #8B0000; border-radius: 5px">
#     <p><u><b>❌ КОММЕНТАРИЙ РЕВЬЮЕРА v.2</b></u></p>
#     <p>Замечание устранено не полностью - при оптимизации моделей необходимо анализировать не accuracy а f1-меру. То есть если значение F1 для текущей итерации цикла, то лучшей модели присваиваются параметры текущей. Пример в ячейке ниже.</p>
# </div>
# Пример фрагмента кода для оптимизации модели
...
    f1_DT = f1_score(prediction_valid_DT, target_valid)
    if best_f1_DT < f1_DT:
        best_DT = DT
        ...
# <div class="alert alert-info" style="border-color: #0080FF; border-radius: 5px">
#         <p><u><b>КОММЕНТАРИЙ СТУДЕНТА</b></u></p>
# спасибо тебе большое за кусок кода, он немного сократил время на раздумование. :)

# <div class="alert alert-success" style="border-color: green; border-radius: 5px">
#     <p><u><b>✅ КОММЕНТАРИЙ РЕВЬЮЕРА v.3</b></u></p>
#     <p>Ок, теперь все верно.</p>
# </div>

# Вывод: Видим низкое значение F1, следовательно низкое качество моделей.

# Функция для отображения соотношения ответов моделей (сколько 0, сколько 1)

# In[30]:


def all_models_share(features_train, target_train, features_valid, target_valid):
    model_DT = DecisionTreeClassifier(random_state = 12345, max_depth = 9)
    model_DT.fit(features_train, target_train)
    DT_share = pd.Series(model_DT.predict(features_valid)).value_counts(normalize = 1)
    
    
    
    model_RF = RandomForestClassifier(random_state = 12345,n_estimators = 40, max_depth = 16)
    model_RF.fit(features_train, target_train)
    RF_share = pd.Series(model_RF.predict(features_valid)).value_counts(normalize = 1)
    
    model_LR = LogisticRegression(random_state = 12345,max_iter= 1000,tol=1e-5,solver = 'lbfgs')
    model_LR.fit(features_train, target_train)
    LR_share = pd.Series(model_LR.predict(features_valid)).value_counts(normalize = 1)
    

    
    print("Доли ответов:")
    print()
    print("Дерево решений", DT_share)
    print()
    print("Случайный лес ", RF_share)
    print()
    print("Логистческая регрессия", LR_share)


# <div class="alert alert-danger" style="border-color: #8B0000; border-radius: 5px">
#     <p><u><b>❌ КОММЕНТАРИЙ РЕВЬЮЕРА v.1</b></u></p>
#     <p>В функции выше используются модели с гиперпараметрами по умолчанию - это неверно.</p>
# </div>

# <div class="alert alert-danger" style="border-color: #8B0000; border-radius: 5px">
#     <p><u><b>❌ КОММЕНТАРИЙ РЕВЬЮЕРА v.2</b></u></p>
#     <p>В функцию all_models_share нужно передать модели с подобранными гиперпараметрами.</p>
# </div>

# <div class="alert alert-success" style="border-color: green; border-radius: 5px">
#     <p><u><b>✅ КОММЕНТАРИЙ РЕВЬЮЕРА v.3</b></u></p>
#     <p>Ок.</p>
# </div>

# Применим функцию отображения соотношения ответов моделей

# In[31]:


all_models_share(features_train, target_train, features_valid, target_valid)


# вывод: Логистичесая регрессия показала болюшую долю положительных ответов, далее случайный лес и в конце дерево решений
# С учетом дисбаланса неудивительно, что модели с большой вероятностью будут выдавать ответ 0, построим матрицы ошибок для моделей

# <div class="alert alert-danger" style="border-color: #8B0000; border-radius: 5px">
#     <p><u><b>❌ КОММЕНТАРИЙ РЕВЬЮЕРА v.1</b></u></p>
#     <p>Вывод в ячейке ниже неверный. Ты же просто определил соотношение 0/1 а не точность моделей.</p>
# </div>

# <div class="alert alert-success" style="border-color: green; border-radius: 5px">
#     <p><u><b>✅ КОММЕНТАРИЙ РЕВЬЮЕРА v.2</b></u></p>
#     <p>Ок!</p>
# </div>

# Матрица ошибок для дерево решений

# In[32]:


confusion_matrix(target_valid, prediction_valid_DT)


# Матрица ошибок для случайный лес

# In[33]:


confusion_matrix(target_valid, prediction_valid_RF)


# Вывод: Матрица показала, что дерево решений склонно выдавать позитивные предсказания, очень высокое количество ложных позитивных предсказания (FP).

# поработаем с логистической регрессией отдельно

# In[34]:


confusion_matrix(target_valid, prediction_valid_LR)


# In[35]:


def plot_roc_curve(fper, tper):
    plt.plot(fper, tper, color='red', label='ROC')
    plt.plot([0, 1], [0, 1], color='green', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC-кривая')
    plt.legend()
    plt.show()


# In[36]:


fper, tper, thresholds = roc_curve(target_valid, LR_probabilities_one_valid)
plot_roc_curve(fper, tper)


# Попробуем обучать логистическую регресию сбалансировав классы

# In[37]:


model_LR = LogisticRegression(random_state = 12345,max_iter= 1000,tol=1e-5,solver = 'lbfgs')
model_LR.fit(features_train, target_train)
LR_probabilities_one_valid_class_weight = model_LR.predict_proba(features_valid)[:, 1]
print("Score", model_LR.score(features_valid, target_valid))
print("AUC-ROC", roc_auc_score(target_valid, LR_probabilities_one_valid_class_weight))

fper, tper, thresholds = roc_curve(target_valid, LR_probabilities_one_valid_class_weight) 
plot_roc_curve(fper, tper)


# Вывод: Отстутсвие улучшений - тоже результат. Высокая точность модели объясняется высокой долей негативных ответов в валидационной выборке.

# <div class="alert alert-danger" style="border-color: #8B0000; border-radius: 5px">
#     <p><u><b>❌ КОММЕНТАРИЙ РЕВЬЮЕРА v.1</b></u></p>
#     <p>Все метрики выше нужно рассчитать для моделей с подобранными гиперпараметрами, а не для дефолтных.</p>
# </div>

# <div class="alert alert-danger" style="border-color: #8B0000; border-radius: 5px">
#     <p><u><b>❌ КОММЕНТАРИЙ РЕВЬЮЕРА v.2</b></u></p>
#     <p>А зачем ты применил параметр class_weight='balanced' к регрессии? В этом пункте мы должны использовать модели без учета дисбаланса.</p>
# </div>

# <div class="alert alert-success" style="border-color: green; border-radius: 5px">
#     <p><u><b>✅ КОММЕНТАРИЙ РЕВЬЮЕРА v.3</b></u></p>
#     <p>Ок!</p>
# </div>

# ## Борьба с дисбалансом

# Как мы выяснили ранее в нашей выборке отрицательны ответов ≈80% , положитительных ≈ 20%. Нам необходмо увеличить количество положительных ответов в 4 раза для достижения баланса. Либо же уменьшить кол-во отрицтаельных ответов.

# Создадим функцию для увеличения представленной класса в выборке 

# In[38]:


def upsample(features, target, repeat, upsampled_сlass):
    """Функция принимаем значение признаков (features[]), целевого признака (target[]), repeat(int / float),
    класс который будет увеличен (upsampled_сlass (0 or 1))"""
    features_zeros = features[target == 0]
    features_ones = features[target == 1]
    target_zeros = target[target == 0]
    target_ones = target[target == 1]
    
    if upsampled_сlass == 0:
        features_upsampled = pd.concat([features_zeros]* repeat + [features_ones] )
        target_upsampled = pd.concat([target_zeros]* repeat + [target_ones] )
        features_upsampled, target_upsampled = shuffle(
        features_upsampled, target_upsampled, random_state=12345)
        
    elif upsampled_сlass == 1:
        features_upsampled = pd.concat([features_zeros] + [features_ones] * repeat)
        target_upsampled = pd.concat([target_zeros] + [target_ones] * repeat)
        features_upsampled, target_upsampled = shuffle(
        features_upsampled, target_upsampled, random_state=12345)
    else:
        features_upsampled = 0
        target_upsampled = 0  
        
        
       
    return features_upsampled, target_upsampled


# Создадим функцию для уменьшения представленной класса в выборке 

# In[39]:


def downsample(features, target, fraction):
    features_zeros = features[target == 0]
    features_ones = features[target == 1]
    target_zeros = target[target == 0]
    target_ones = target[target == 1]

    features_downsampled = pd.concat(
        [features_zeros.sample(frac=fraction, random_state=12345)] + [features_ones])
    target_downsampled = pd.concat(
        [target_zeros.sample(frac=fraction, random_state=12345)] + [target_ones])
    
    features_downsampled, target_downsampled = shuffle(
        features_downsampled, target_downsampled, random_state=12345)
    
    return features_downsampled, target_downsampled


# Применим функцию upsample.
# увеличим количество положительных ответов в 4 раза.
# Протестируем функцию.

# In[40]:


features_train_upsampled, target_train_upsampled = upsample(features_train, target_train, 4, 1)
print(target_train_upsampled.value_counts(normalize = 1))
print(target_train_upsampled.shape)


# <div class="alert alert-success" style="border-color: green; border-radius: 5px">
#     <p><u><b>✅ КОММЕНТАРИЙ РЕВЬЮЕРА v.1</b></u></p>
#     <p>Функция есть! Коэффициент repeat подобран верно, молодец.</p>
#     <p>Круто, что  проверил результат работы функции методом <code>.shape</code></p>
# </div>

# Применим функцию upsample 
# увеличим количество положительных ответов в 4 раза

# In[41]:


features_train_upsampled, target_train_upsampled = upsample(features_train, target_train, 4, 1)
print(target_train_upsampled.value_counts(normalize = 1))
print(target_train_upsampled.shape)


# Применим функцию downsample.
# Уменьшим кол-в пооложительных ответов в 4 раза.
# Протестируем функцию.

# In[42]:


features_downsampled_train, target_downsampled_train = downsample(features_train, target_train, 0.25)
print(target_downsampled_train.value_counts(normalize = 1))
print(target_downsampled_train.shape)


# <div class="alert alert-danger" style="border-color: #8B0000; border-radius: 5px">
#     <p><u><b>❌ КОММЕНТАРИЙ РЕВЬЮЕРА v.1</b></u></p>
#     <p>К сожалению, параметр fraction подобран неверно, ведь соотношение классов 1/4, а не 1/5.</p>
# </div>

# <div class="alert alert-danger" style="border-color: #8B0000; border-radius: 5px">
#     <p><u><b>❌ КОММЕНТАРИЙ РЕВЬЮЕРА v.2</b></u></p>
#     <p>Параметр fraction должен быть равен 0.25</p>
# </div>

# <div class="alert alert-success" style="border-color: green; border-radius: 5px">
#     <p><u><b>✅ КОММЕНТАРИЙ РЕВЬЮЕРА v.3</b></u></p>
#     <p>Ок.</p>
# </div>

# In[43]:


target_train_upsampled.plot(kind ='hist', bins=3, figsize=(5,5))


# In[44]:


target_downsampled_train.plot(kind ='hist', bins=3, figsize=(5,5))


# вывод:видим,что функции выдали одинаковый результат. Есть смысл проверить обе

# проверим функцию 'upsampled'

# Решающее дерево

# In[45]:


best_DT_upsampled = None
best_accuracy_DT_upsampled = 0
best_f1_DT_upsampled=0
best_depth_DT_upsampled = 0
for depth in tqdm(range(2,52)):
    DT_upsampled = DecisionTreeClassifier(random_state = 12345, max_depth = depth)
    DT_upsampled.fit(features_train_upsampled, target_train_upsampled)
    prediction_valid_DT_upsampled = DT_upsampled.predict(features_valid)
    accuracy_DT_upsampled = accuracy_score(prediction_valid_DT_upsampled, target_valid)
    f1_DT_upsampled = f1_score(prediction_valid_DT_upsampled, target_valid)
    if best_f1_DT_upsampled < f1_DT_upsampled:
        best_f1_DT_upsampled = f1_DT_upsampled
        best_DT_upsampled = DT_upsampled
        best_accuracy_DT_upsampled = accuracy_DT_upsampled
        best_depth_DT_upsampled = depth
        
DT_upsampled_probabilities_one_valid = best_DT_upsampled.predict_proba(features_valid)[:, 1]

print(rec_prec_f1(target_valid,prediction_valid_DT_upsampled))
print('AUC-ROC',roc_auc_score(target_valid, DT_upsampled_probabilities_one_valid))
print()
print('лучшая модель с параметрами:', 'глубина-',best_depth_DT_upsampled)


# Случайный лес

# In[46]:


best_RF_upsampled = None
best_accuracy_RF_upsampled = 0
best_f1_RF_upsampled = 0
best_est_RF_upsampled = 0
best_depth_RF_upsampled = 0
for est in tqdm(range(10,200,10)):
    for depth in range(2,25):
        RF_upsampled = RandomForestClassifier(random_state = 12345,n_estimators = est, max_depth = depth)
        RF_upsampled.fit(features_train_upsampled, target_train_upsampled)
        prediction_valid_RF_upsampled = RF_upsampled.predict(features_valid)
        accuracy_RF_upsampled = accuracy_score(prediction_valid_RF_upsampled, target_valid)
        f1_RF_upsampled = f1_score(prediction_valid_RF_upsampled, target_valid)
        if best_f1_RF_upsampled < f1_RF_upsampled:
            best_f1_RF_upsampled = f1_RF_upsampled
            best_RF_upsampled = RF_upsampled
            best_depth_RF_upsampled = depth
            best_est_RF_upsampled = est
            best_accuracy_RF_upsampled = accuracy_RF_upsampled
            

RF_upsampled_probabilities_one_valid = best_RF_upsampled.predict_proba(features_valid)[:, 1]

print(rec_prec_f1(target_valid,prediction_valid_RF_upsampled))
print('AUC-ROC',roc_auc_score(target_valid, RF_upsampled_probabilities_one_valid))
print()
print('лучшая модель с параметрами:', 'глубина-',best_depth_RF_upsampled,'','количество ветвей-',best_est_RF_upsampled)


# Логистическая регрессия

# In[47]:


best_LR_upsampled = None
best_f1_LR_upsampled=0
best_accuracy_LR_upsampled = 0
for est in tqdm(range(2,52)):
    LR_upsampled = LogisticRegression(random_state = 12345,max_iter= 1000,tol=1e-5,solver = 'liblinear' )
    LR_upsampled.fit(features_train_upsampled, target_train_upsampled)
    LR_prediction_upsampled = LR_upsampled.predict(features_valid)
    accuracy_LR_upsampled = accuracy_score(LR_prediction_upsampled, target_valid)
    f1_LR_upsampled = f1_score(LR_prediction_upsampled, target_valid)
    if best_f1_LR_upsampled < f1_LR_upsampled:
        best_f1_LR_upsampled = f1_LR_upsampled
        best_LR_upsampled = LR_upsampled
        best_accuracy_LR_upsampled = accuracy_LR_upsampled
            
LR_upsampled_probabilities_one_valid = best_LR_upsampled.predict_proba(features_valid)[:, 1]            
            
print(rec_prec_f1(target_valid, LR_prediction_upsampled))
print('AUC-ROC',roc_auc_score(target_valid, LR_upsampled_probabilities_one_valid))


# <div class="alert alert-danger" style="border-color: #8B0000; border-radius: 5px">
#     <p><u><b>❌ КОММЕНТАРИЙ РЕВЬЮЕРА v.1</b></u></p>
#     <p>Для борьбы с дисбалансом нам знакомы три метода Upsampling, Downsampling и Взвешиваие классов (class_weight='balanced')</p>
#     <p>Применение class_weight='balanced' при Upsampling и Downsampling не влияет на результаты, поскольку это самомстоятельный метод. class_weight='balanced' нужно проводить на стандартных выборках.</p> 
# </div>

# <div class="alert alert-danger" style="border-color: #8B0000; border-radius: 5px">
#     <p><u><b>❌ КОММЕНТАРИЙ РЕВЬЮЕРА v.2</b></u></p>
#     <p>Здесь аналогичное замечание - при оптимизации моделей необходимо анализировать не accuracy а f1-меру. То есть если значение F1 для текущей итерации цикла, то лучшей модели присваиваются параметры текущей. Пример в ячейке ниже.</p>
# </div>

# <div class="alert alert-success" style="border-color: green; border-radius: 5px">
#     <p><u><b>✅ КОММЕНТАРИЙ РЕВЬЮЕРА v.3</b></u></p>
#     <p>Ок!</p>
# </div>

# проверим метод 'downsample'

# Решающее дерево

# In[48]:


best_DT_downsampled = None
best_accuracy_DT_downsampled = 0
best_f1_DT_downsampled=0
best_depth_DT_downsampled = 0
for depth in tqdm(range(2,52)):
    DT_downsampled = DecisionTreeClassifier(random_state = 12345, max_depth = depth)
    DT_downsampled.fit(features_downsampled_train, target_downsampled_train)
    prediction_valid_DT_downsampled = DT_downsampled.predict(features_valid)
    accuracy_DT_downsampled = accuracy_score(prediction_valid_DT_downsampled, target_valid)
    f1_DT_downsampled= f1_score(prediction_valid_DT_downsampled, target_valid)
    if best_f1_DT_downsampled < f1_DT_downsampled:
        best_f1_DT_downsampled = f1_DT_downsampled
        best_DT_downsampled = DT_downsampled
        best_accuracy_DT_downsampled = accuracy_DT_downsampled
        best_depth_DT_downsampled = depth
        
DT_downsampled_probabilities_one_valid = best_DT_downsampled.predict_proba(features_valid)[:, 1]

print(rec_prec_f1(target_valid,prediction_valid_DT_downsampled))
print('AUC-ROC',roc_auc_score(target_valid, DT_downsampled_probabilities_one_valid))
print()
print('лучшая модель с параметрами:', 'глубина-',best_depth_DT_downsampled)


# Случайный лес

# In[49]:


best_RF_downsampled = None
best_accuracy_RF_downsampled = 0
best_f1_RF_downsampled=0
best_est_RF_downsampled = 0
best_depth_RF_downsampled = 0
for est in tqdm(range(10,200,10)):
    for depth in range(2,25):
        RF_downsampled = RandomForestClassifier(random_state = 12345,n_estimators = est, max_depth = depth)
        RF_downsampled.fit(features_downsampled_train, target_downsampled_train)
        prediction_valid_RF_downsampled = RF_downsampled.predict(features_valid)
        accuracy_RF_downsampled = accuracy_score(prediction_valid_RF_downsampled, target_valid)
        f1_RF_downsampled = f1_score(prediction_valid_RF_downsampled, target_valid)
        if best_f1_RF_downsampled < f1_RF_downsampled:
            best_f1_RF_downsampled = f1_RF_downsampled
            best_RF_downsampled = RF_downsampled
            best_depth_RF_downsampled = depth
            best_est_RF_downsampled = est
            best_accuracy_RF_downsampled = accuracy_RF_downsampled
            

RF_downsampled_probabilities_one_valid = best_RF_downsampled.predict_proba(features_valid)[:, 1]

print(rec_prec_f1(target_valid,prediction_valid_RF_downsampled))
print('AUC-ROC',roc_auc_score(target_valid, RF_downsampled_probabilities_one_valid))
print()
print('лучшая модель с параметрами:', 'глубина-',best_depth_RF_downsampled,'','количество ветвей-',best_est_RF_downsampled)


# Логистическая регрессия

# In[50]:


best_LR_downsampled = None
best_f1_LR_downsampled=0
best_accuracy_LR_downsampled = 0
for est in tqdm(range(2,52)):
    LR_downsampled = LogisticRegression(random_state = 12345,max_iter= 1000,tol=1e-5,solver = 'liblinear' )
    LR_downsampled.fit(features_downsampled_train, target_downsampled_train)
    LR_prediction_downsampled = LR_downsampled.predict(features_valid)
    accuracy_LR_downsampled = accuracy_score(LR_prediction_downsampled, target_valid)
    f1_LR_downsampled = f1_score(LR_prediction_downsampled, target_valid)
    if best_f1_LR_downsampled < f1_LR_downsampled:
        best_f1_LR_downsampled = f1_LR_downsampled
        best_LR_downsampled = LR_downsampled
        best_accuracy_LR_downsampled = accuracy_LR_downsampled
            
LR_downsampled_probabilities_one_valid = best_LR_downsampled.predict_proba(features_valid)[:, 1]            
            
print(rec_prec_f1(target_valid, LR_prediction_downsampled))
print('AUC-ROC',roc_auc_score(target_valid, LR_downsampled_probabilities_one_valid))


# проверим Точность моделей на выборке с дисбалансом

# In[51]:


print(best_accuracy_RF)
print(best_accuracy_DT)
print(best_accuracy_LR)


# проверим Точность моделей на сбалансированной увеличенной выборке

# In[52]:


print(best_accuracy_RF_upsampled)
print(best_accuracy_DT_upsampled)
print(best_accuracy_LR_upsampled)


# проверим Точность моделей на сбалансированной уменьшенной выборке

# In[53]:


print(best_accuracy_RF_downsampled)
print(best_accuracy_DT_downsampled)
print(best_accuracy_LR_downsampled)


# <div class="alert alert-danger" style="border-color: #8B0000; border-radius: 5px">
#     <p><u><b>❌ КОММЕНТАРИЙ РЕВЬЮЕРА v.1</b></u></p>
#     <p>Здесь для качественного эксперимента тоже стоит заново подобрать гиперпараметры для всех моделей, для всех медодов борьбы с дисбалансом.</p>
#     <p>При подборе гиперпараметров ориентироваться на метрику f1, дополнительно рассчитать метрику AUC-ROC.</p>
# </div>

# Вывод:Показаели всех моделей  обученных на сбалонсированных выборках ухудшились.Для 'upsampled' не сильно значительно, а вот для 'downsampled' ухудшения оказались более значительными.Поэтому я буду использовать метод 'upsampled' далее по проекту.
# 
# Лучшие результаты показывает модель 'RandomForest' обученная на данных сбалансированых методом 'upsampled'. На валидационной выборке 'RandomForest' уже показывает резульаты F1 меры = 0.60, что удовлетваряет условию данного проекта.

# <div class="alert alert-danger" style="border-color: #8B0000; border-radius: 5px">
#     <p><u><b>❌ КОММЕНТАРИЙ РЕВЬЮЕРА v.2</b></u></p>
#     <p>В своей работе ты провел эксеприменты с методом Upsampling. Давай проведем такие же эксперименты еще с одинм из методов на твой выбор (Взвешивание классов или Downsampling). Тогда проект будет полным.</p>
# </div>

# <div class="alert alert-success" style="border-color: green; border-radius: 5px">
#     <p><u><b>✅ КОММЕНТАРИЙ РЕВЬЮЕРА v.3</b></u></p>
#     <p>Ок!</p>
# </div>

# ## Тестирование модели

# Обучим финальную модель с обьединенными данными

# In[54]:


features_full_train = pd.concat([features_train, features_valid])
target_full_train = pd.concat([target_train, target_valid])


# In[55]:


features_upsampled, target_upsampled = upsample(features_full_train, target_full_train, 4, 1)


# In[61]:


RF_final_full_data = RandomForestClassifier(random_state = 12345,n_estimators = 150, max_depth = 13)
RF_final_full_data.fit(features_upsampled, target_upsampled)

prediction_valid_RF_final_full_data = RF_final_full_data.predict(features_test)

f1_RF_final_full_data= f1_score(prediction_valid_RF_final_full_data, target_test)

RF_full_data_probabilities_one_valid = RF_final_full_data.predict_proba(features_test)[:, 1]

print(rec_prec_f1(target_test,prediction_valid_RF_final_full_data))
print('AUC-ROC',roc_auc_score(target_test, RF_full_data_probabilities_one_valid))


# Обучим финальную модель без обьединенных данных

# In[57]:


RF_final = RandomForestClassifier( max_depth= 12,  n_estimators = 20, random_state=12345)
RF_final.fit(features_train_upsampled, target_train_upsampled)

RF_final_prediction =RF_final.predict(features_test)

RF_final_valid = RF_final.predict_proba(features_test)[:, 1]

auc_roc_RF = roc_auc_score(target_test, RF_final_valid)

print(rec_prec_f1(target_test, RF_final_prediction))
print('AUC-ROC',roc_auc_score(target_test, RF_final_valid))


# вывод: модель обученная на обьедененных данных оказалась менее точной (даже после маштобирования данных) по F-1 мере, а вот AUC-ROC у нее оказался выше,но мы в первую очередь смотрим на F-1 меру по этому мы будем исполоьзовать модель обученную на не обьеденненых данных.

# Создаем константную модель

# In[58]:


target_predict_constant = pd.Series([0]*len(target_test))
target_predict_constant.value_counts()


# Сравним показатель точности (accuracy_score) константной модели и финальной

# In[59]:


print('accuracy_score константой модели:', accuracy_score(target_valid, target_predict_constant))
print('accuracy_score финальной модели:', accuracy_score(target_test, RF_final_prediction))


# Дополнительно сравним AUC-ROC — единственный параметр подающийся сравнению, потому что константная подель содержит только негативные ответы

# In[60]:


print('AUC-ROC константой модели:', roc_auc_score(target_valid, target_predict_constant))
print('AUC-ROC финальной модели:', roc_auc_score(target_test, RF_final_valid))


# вывод:Финальная модель показывает результаты лучше, чем константная модель — модель можно считать адекватной.

# <div class="alert alert-danger" style="border-color: #8B0000; border-radius: 5px">
#     <p><u><b>❌ КОММЕНТАРИЙ РЕВЬЮЕРА v.1</b></u></p>
#     <p>А здесь необходимо использовать лучшую модель и лучший метод борьбы с дисбалансом, из тех что ты исследовал в пункте борьба с дисбалансом. При этом лучшую модель нужно обучить на объединенной выборке трейн+валидация.</p> 
# </div>

# <div class="alert alert-warning" style="border-color: orange; border-radius: 5px">
#     <p><u><b>⚠️ КОММЕНТАРИЙ РЕВЬЮЕРА v.2</b></u></p>
#     <p>А здесь есть важный момент. Помнишь, у нас есть валидационная выборка? Давай объединим ее с обучающей выборкой. Так мы получим большее количество данных! Используем лучшую модель с гиперпараметрами, которые ты подберешь для нее на этапе борьбы с дисбалансом, и заново обучим модель.</p>
#     <p>После этого проведем тестирование. Вдруг метрики будут лучше? Ниже приведу прмер объединения выборок.</p>
# </div>
# Код ревьюера
features_full_train = pd.concat([features_train, features_valid])
target_full_train = pd.concat([target_train, target_valid])

# При необходимости можно увеличить выборку
features_upsampled, target_upsampled = upsample(features_full_train, target_full_train, 4)

# ... или уменьшить
features_downsampled, target_downsampled = downsample(features_full_train, target_full_train, 0.25)
# <div class="alert alert-danger" style="border-color: #8B0000; border-radius: 5px">
#     <p><u><b>❌ КОММЕНТАРИЙ РЕВЬЮЕРА v.3</b></u></p>
#     <p>Молодец, что попробовал объединить выборки. Это очень эффективный способ.</p>
#     <p>Только в этом пункте уже не нужно подбирать гиперпараметры в цикле. Достаточно просто использовать модель с лучшими из тех, что ты подобрал ранее. В нашем случае это RF_upsampled = RandomForestClassifier(random_state = 12345,n_estimators = 150, max_depth = 13) именно эта модель дала наивысшую метрику f1. Ее нужно обучить на объединенной увеличенной выборке и протестировать на тестовой. После этого не забудь исправить вывод.</p>
# </div>

# <div class="alert alert-success" style="border-color: green; border-radius: 5px">
#     <p><u><b>✅ КОММЕНТАРИЙ РЕВЬЮЕРА v.4</b></u></p>
#     <p>Ок!</p>
# </div>

# <div class="alert alert-danger" style="border-color: #8B0000; border-radius: 5px">
#     <p><u><b>❌ КОММЕНТАРИЙ РЕВЬЮЕРА v.1</b></u></p>
#     <p><b>Общее замечание к проекту</b></p>
#     <p>Комментарии к проекту нужно оформлять в ячейках типа Markdown вместо # закомментированных строк.</p>
#     <p>Также нужно везде исправить упавший код.</p>
# </div>

# <div class="alert alert-success" style="border-color: green; border-radius: 5px">
#     <p><u><b>✅ КОММЕНТАРИЙ РЕВЬЮЕРА v.2</b></u></p>
#     <p>Ок!</p>
# </div>

# ## Вывод

# Итоговый вывод:В первоначальные данных наблюдался значительный дисбаланс (80% ответов целевого признака были негативными и только 20% позитивными), из-за чего обученная на этих данных модель не проходила проверку на адекватность. Все модели не первоначальных данных характеризовались высокой степенью ошибок и низким качеством взвешенной величины (F1) — модели показывали низкие результаты точности и полноты.
# 
# Мы устранили дисбаланс классов в обучающей выборки методом upsampling — увеличили количество значений позитивного класса в 4 раза. Так мы достигли баланса классов обучаюющей выборки: 0 - 0.501043 1 - 0.498957.
# 
# также использовали метод downsampling- уменьшили количество значений позитивного класса в 4 раза.Получили такой же результат как и у метода upsampling: 0 - 0.501043 1 - 0.498957.НО этот метод сильно ухудшает качество моделей из-за недостатка данных для обучения.По этой причине в дальнейшем был использован метод upsampling ктоторый ухудшает качество модели не столь критично.
# 
# Разобрали несколько вариантов борьбы с дисбалансом upsampling и downsampling
# 
# На новых данных все модели показали результат выше, чем на несбалансированной выборке. Лучшие показатели были у модели случайного леса:
# 
# Полнота 0.5515151515151515
# Точность 0.674074074074074
# F1-мера 0.6066666666666666
# AUC-ROC 0.8339155332856119
# лучшая модель с параметрами: глубина- 12  количество ветвей- 20 и рандомное состояне- 12345
# 
# была расмотренная финальная модель с обьеденненой валидационной и тренеровочной выборками, она показала результат хуже чем модель обученная на необьеденненой маштабированной быборке, по этой причине была отвергнута как не подходящая по значению F-1 меры.
# Финальная модель прошла проверку на адекватность в сравнении с контантной моделью: accuracy_score константой модели: 0.85
# accuracy_score константой модели: 0.791
# accuracy_score финальной модели: 0.837
# AUC-ROC константой модели: 0.5
# AUC-ROC финальной модели: 0.8495744830760144
# модель с параметрами: глубина- 12  количество ветвей- 20 и рандомное состояне- 12345

# <div class="alert alert-danger" style="border-color: #8B0000; border-radius: 5px">
#     <p><u><b>❌ КОММЕНТАРИЙ РЕВЬЮЕРА v.1</b></u></p>
#     <p>Давай здесь напишем вывод о проделанной работе по плану "исходнные данные - что следали - что получили".</p>
# </div>

# <div class="alert alert-warning" style="border-color: orange; border-radius: 5px">
#     <p><u><b>⚠️ КОММЕНТАРИЙ РЕВЬЮЕРА v.2</b></u></p>
#     <p>Отлично вывод на месте. Не забудь откорректировать его после внесения изменений в проект.</p>
# </div>

# ## Чек-лист готовности проекта

# Поставьте 'x' в выполненных пунктах. Далее нажмите Shift+Enter.

# - [x]  Jupyter Notebook открыт
# - [x]  Весь код выполняется без ошибок
# - [x]  Ячейки с кодом расположены в порядке исполнения
# - [x]  Выполнен шаг 1: данные подготовлены
# - [x]  Выполнен шаг 2: задача исследована
#     - [x]  Исследован баланс классов
#     - [x]  Изучены модели без учёта дисбаланса
#     - [x]  Написаны выводы по результатам исследования
# - [x]  Выполнен шаг 3: учтён дисбаланс
#     - [x]  Применено несколько способов борьбы с дисбалансом
#     - [x]  Написаны выводы по результатам исследования
# - [x]  Выполнен шаг 4: проведено тестирование
# - [x]  Удалось достичь *F1*-меры не менее 0.59
# - [x]  Исследована метрика *AUC-ROC*

# ## Финальные комментарии ревьюера

# <div class="alert alert-danger" style="border-color: #8B0000; border-radius: 5px">
#     <p><u><b>❌ КОММЕНТАРИЙ РЕВЬЮЕРА v.1</b></u></p>
#     <p>Ратимир, я проверил твой проект, он хороший, крепкий. Осталось немного. Думаю, тебе не составит труда исправить замечания и вернуть мне тетредку на повторную проверку! Все мои комментарии ты найдешь по ходу проекта. Если возникнут вопросы задавай.</p>
#     <p>Успехов!</p>
# </div>

# <div class="alert alert-danger" style="border-color: #8B0000; border-radius: 5px">
#     <p><u><b>❌ КОММЕНТАРИЙ РЕВЬЮЕРА v.2</b></u></p>
#     <p>Ратимир, эта работа уже близка к зачету. Осталось немного. Поправь, пожалуйста недочеты, указанные в моих комментариях v.2</p>
# </div>

# <div class="alert alert-danger" style="border-color: #8B0000; border-radius: 5px">
#     <p><u><b>❌ КОММЕНТАРИЙ РЕВЬЮЕРА v.3</b></u></p>
#     <p>Ратимир, осталось поправить пункт с тестированием. Поправь, пожалуйста недочеты, указанные в моих комментариях v.3</p>
# </div>

# <div class="alert alert-success" style="border-color: green; border-radius: 5px">
#     <p><u><b>✅ КОММЕНТАРИЙ РЕВЬЮЕРА v.4</b></u></p>
#     <p>Ратимир, мы добрались до зачета по всем пунктам, замечаний больше нет. Ты молодец!</p>
#     <p>Надеюсь тебя посетило то классное чувство профессиональной прокачки по ходу решения.</p>
#     <p>Для систематизации знаний делюсь с тобой ссылкой на учебник по ML от ШАД:</p>
# 
# https://ml-handbook.ru/</p>
# 
# <p>Тут много полезного по метематике:</p>
# 
# http://mathprofi.ru/</p>
# 
# <p>Очень классный бесплатный тренажер по SQL (SQL вы будуте проходить дальше в курсе, он крайне важен для работы, так что лучше начать изучать его заранее):</p>
# 
# https://sql-ex.ru/</p>
# 
# <p>Настоятельно рекомендую пройти серию курсов по статистике на Stepik:</p>
# 
# https://stepik.org/course/76/syllabus</p>
# 
# <p>Там же на Stepik очень много полезных бесплатных курсов по Python, Ананлизу данных и DS.</p>
# <p>Желаю не останавливаться на пути к поставленной цели. Все получится!</p>
# </div>

# In[ ]:




