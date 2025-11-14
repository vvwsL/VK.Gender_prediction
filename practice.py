import pandas as pd
import numpy as np
import ast
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
import joblib

# Пути к исходным файлам данных (замените на свои пути)
train_csv_path = 'train.csv'             # Данные для обучения
train_labels_path = 'train_labels.csv'   # Метки пола пользователей для обучения
test_csv_path = 'test.csv'                # Данные для теста
test_users_path = 'test_users.csv'       # Список пользователей для теста
geo_info_path = 'geo_info.csv'            # Информация по геолокации
referer_vectors_path = 'referer_vectors.csv'  # Векторные признаки для referer

# Файлы для сохранения модели и результатов
model_file = 'model_prediction_gender.pkl'
train_features_file = 'train_features.csv'
submission_file = 'gender_predictions.csv'


# Функция для разбора строки user_agent в словарь, длинная строчка с большим количеством информации
def parse_user_agent(ua_str):
    try:
        return ast.literal_eval(ua_str)
    except:
        return {}

# Функция для расширения колонки user_agent_parsed в несколько отдельных колонок
def expand_user_agent(df):
    ua_df = pd.json_normalize(df['user_agent_parsed'])
    return pd.concat([df.drop(columns=['user_agent_parsed']), ua_df], axis=1)

# Функция для выделения топ-N категорий, остальные попадут в категорию 'other'
def top_category(series, n=10):
    top_cats = series.value_counts().nlargest(n).index.tolist()  # Берем топ-N наиболее частых категорий
    # Заменяем остальные значения на 'other'
    return series.where(series.isin(top_cats), other='other').astype('category')

# Регулярное выражение для разбора URL referer на домен и путь
pattern = r'https://([^/]+)/?(.*)'

# Список колонок, которые будем рассматривать как категориальные
categorical_cols = ['user_id', 'referer_domain_top', 'referer_path', 'geo_id', 'country_id', 'region_id', 'timezone', 'browser_top', 'os_top']

# Загружаем основные таблицы с данными для обучения
train_df = pd.read_csv(train_csv_path, sep=';', encoding='utf-8')
train_labels_df = pd.read_csv(train_labels_path, sep=';', encoding='utf-8')
geo_info_df = pd.read_csv(geo_info_path, sep=';', encoding='utf-8')
referer_vectors_df = pd.read_csv(referer_vectors_path, sep=';', encoding='utf-8')

# Объединяем данные train 
train_df['user_id'] = train_df['user_id'].astype('category')
train_labels_df['user_id'] = train_labels_df['user_id'].astype('category')
train_df.set_index('user_id', inplace=True)
train_labels_df.set_index('user_id', inplace=True)
train_full_df = train_df.join(train_labels_df, how='left').reset_index()

# Разбор user_agent в словарь и расширение на отдельные признаки
train_full_df['user_agent_parsed'] = train_full_df['user_agent'].apply(parse_user_agent)
train_full_df = expand_user_agent(train_full_df)

# Извлечение из referer домена и пути по заданному шаблону
train_full_df[['referer_domain', 'referer_path']] = train_full_df['referer'].str.extract(pattern)
train_full_df['referer_domain'] = train_full_df['referer_domain'].fillna('unknown')
train_full_df['referer_path'] = train_full_df['referer_path'].fillna('unknown')

# Ограничиваем количество уникальных значений, а все остальные объединяем в 'other'
train_full_df['referer_domain_top'] = top_category(train_full_df['referer_domain'])
train_full_df['browser'] = train_full_df['browser'].fillna('unknown')
train_full_df['browser_top'] = top_category(train_full_df['browser'])
train_full_df['os'] = train_full_df['os'].fillna('unknown')
train_full_df['os_top'] = top_category(train_full_df['os'])

# Объединяем с дополнительными признаками из referer_vectors и geo_info
train_full_df = train_full_df.merge(referer_vectors_df, on='referer', how='left')
train_full_df['geo_id'] = train_full_df['geo_id'].astype(str)
geo_info_df['geo_id'] = geo_info_df['geo_id'].astype(str)
train_full_df = train_full_df.merge(geo_info_df, on='geo_id', how='left')

# Отмечаем выбранные колонки как категориальные
for c in categorical_cols:
    if c in train_full_df.columns:
        train_full_df[c] = train_full_df[c].astype('category')

# Создаем временные признаки из времени запроса
train_full_df['request_ts'] = pd.to_numeric(train_full_df['request_ts'], errors='coerce')  # Время запроса в секундах
train_full_df['dt'] = pd.to_datetime(train_full_df['request_ts'], unit='s', errors='coerce')  # Преобразуем в datetime
train_full_df['hour'] = train_full_df['dt'].dt.hour         # Час из времени запроса
train_full_df['weekday'] = train_full_df['dt'].dt.weekday   # День недели из времени запроса (0 - это понедельник)

# Считаем количество запросов на каждого пользователя
requests_count = train_full_df.groupby('user_id').size().rename('requests_count')

# Расчитываем относительную частоту для топовых referer доменов
filtered_domains = train_full_df.dropna(subset=['user_id', 'referer_domain_top'])
domain_counts = pd.crosstab(filtered_domains['user_id'], filtered_domains['referer_domain_top'], dropna=False)
domain_freq = domain_counts.div(domain_counts.sum(axis=1), axis=0).fillna(0)
domain_freq.columns = [f'domain_freq_{col}' for col in domain_freq.columns]

# Расчитываем относительную частоту для браузеров
filtered_browsers = train_full_df.dropna(subset=['user_id', 'browser_top'])
browser_counts = pd.crosstab(filtered_browsers['user_id'], filtered_browsers['browser_top'], dropna=False)
browser_freq = browser_counts.div(browser_counts.sum(axis=1), axis=0).fillna(0)
browser_freq.columns = [f'browser_freq_{col}' for col in browser_freq.columns]

# Расчитываем относительную частоту для ОС
filtered_oses = train_full_df.dropna(subset=['user_id', 'os_top'])
os_counts = pd.crosstab(filtered_oses['user_id'], filtered_oses['os_top'], dropna=False)
os_freq = os_counts.div(os_counts.sum(axis=1), axis=0).fillna(0)
os_freq.columns = [f'os_freq_{col}' for col in os_freq.columns]

# Итоговые колонки для компонентов, если они есть
component_cols = [f'component{i}' for i in range(10) if f'component{i}' in train_full_df.columns]

# Вычисляем статистики по компонентам для каждого пользователя
comp_agg = train_full_df.groupby('user_id')[component_cols].agg(['mean','median','max','min','std'])
comp_agg.columns = ['_'.join(col) for col in comp_agg.columns]

# Вычисляем статистики по времени запросов
time_agg = train_full_df.groupby('user_id')['request_ts'].agg(['mean','min','max'])
time_agg['range'] = time_agg['max'] - time_agg['min']

# Частоты по часам запроса
hour_counts = pd.crosstab(train_full_df['user_id'], train_full_df['hour'])
hour_freq = hour_counts.div(hour_counts.sum(axis=1), axis=0).fillna(0)
hour_freq.columns = [f'hour_freq_{h}' for h in hour_freq.columns]

# Частоты по дням недели
weekday_counts = pd.crosstab(train_full_df['user_id'], train_full_df['weekday'])
weekday_freq = weekday_counts.div(weekday_counts.sum(axis=1), axis=0).fillna(0)
weekday_freq.columns = [f'weekday_freq_{d}' for d in weekday_freq.columns]

# Собираем все признаки в один датафрейм
features = pd.concat([
    requests_count,
    domain_freq,
    browser_freq,
    os_freq,
    comp_agg,
    time_agg,
    hour_freq,
    weekday_freq
], axis=1).fillna(0)

# Добавляем целевую переменную для каждого пользователя
user_target_map = train_full_df.drop_duplicates(subset=['user_id']).set_index('user_id')['target']
features['target'] = features.index.map(user_target_map)

# Оставляем только строки с известныи значением переменной для обучения
train_features = features[features['target'].notna()].copy()

# Формируем матрицу признаков и целевой вектор
X = train_features.drop(columns=['target'])
y = train_features['target'].astype(int)

# Делим выборку на обучающую и валидационную (stratify для равномерного распределения пола)
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# Создаем датасеты для LightGBM
train_data = lgb.Dataset(X_train, label=y_train)
val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

# Параметры модели LightGBM
params = {
    'objective': 'binary',               # Задача бинарной классификации
    'metric': ['binary_logloss','auc'], # Метрики для оценки
    'boosting_type': 'gbdt',
    'verbosity': 1,
    'seed': 42,
    'learning_rate': 0.05,
    'num_leaves': 31,
    'max_depth': -1,
    'feature_fraction': 0.8,             # Использование 80% признаков на каждой итерации
    'bagging_fraction': 0.8,             # Использование 80% данных на каждой итерации
    'bagging_freq': 5,
}

# Обучение модели
model = lgb.train(
    params,
    train_data,
    num_boost_round=1000,
    valid_sets=[train_data, val_data],
    callbacks=[lgb.early_stopping(stopping_rounds=50), lgb.log_evaluation(period=50)]
)

# Предсказания вероятности пола на валидационной выборке
y_pred_prob = model.predict(X_val, num_iteration=model.best_iteration)
# Преобразуем вероятность в бинарную метку (0 или 1), порог 0.5
y_pred = (y_pred_prob > 0.5).astype(int)

# Оценка качества модели
print(f"Accuracy на валидации: {accuracy_score(y_val, y_pred):.4f}")
print(f"ROC AUC на валидации: {roc_auc_score(y_val, y_pred_prob):.4f}")

# Сохраняем модель и обученные признаки
joblib.dump(model, model_file)
train_features.drop(columns=['target']).to_csv(train_features_file)
print("Модель и тренировочные признаки сохранены.")

# Загружаем данные теста и вспомогательные файлы
test_users_df = pd.read_csv(test_users_path, sep=';', encoding='utf-8')
test_df = pd.read_csv(test_csv_path, sep=';', encoding='utf-8')
geo_info_df = pd.read_csv(geo_info_path, sep=';', encoding='utf-8')
referer_vectors_df = pd.read_csv(referer_vectors_path, sep=';', encoding='utf-8')

# Оставляем в тестовых данных только нужных пользователей
test_df = test_df[test_df['user_id'].isin(test_users_df['user_id'])]

# Парсим user_agent по тому же методу
test_df['user_agent_parsed'] = test_df['user_agent'].apply(parse_user_agent)
test_df = expand_user_agent(test_df)

# Разбираем referer на домен и путь
test_df[['referer_domain', 'referer_path']] = test_df['referer'].str.extract(pattern)
test_df['referer_domain'] = test_df['referer_domain'].fillna('unknown')
test_df['referer_path'] = test_df['referer_path'].fillna('unknown')

# Используем категории из тренировочных данных, чтоб избежать рассогласования
train_ref_dom_cats = train_full_df['referer_domain_top'].cat.categories if 'referer_domain_top' in train_full_df else []
train_browser_cats = train_full_df['browser_top'].cat.categories if 'browser_top' in train_full_df else []
train_os_cats = train_full_df['os_top'].cat.categories if 'os_top' in train_full_df else []

# Ограничиваем категории в тесте по учебным категориям и 'other'
test_df['referer_domain_top'] = test_df['referer_domain'].where(test_df['referer_domain'].isin(train_ref_dom_cats), other='other').astype('category')
test_df['browser'] = test_df['browser'].fillna('unknown')
test_df['browser_top'] = test_df['browser'].where(test_df['browser'].isin(train_browser_cats), other='other').astype('category')
test_df['os'] = test_df['os'].fillna('unknown')
test_df['os_top'] = test_df['os'].where(test_df['os'].isin(train_os_cats), other='other').astype('category')

# Объединяем с дополнительными признаками
test_df = test_df.merge(referer_vectors_df, on='referer', how='left')

test_df['geo_id'] = test_df['geo_id'].astype(str)
geo_info_df['geo_id'] = geo_info_df['geo_id'].astype(str)
test_df = test_df.merge(geo_info_df, on='geo_id', how='left')

# Категоризируем колонки
for col in categorical_cols:
    if col in test_df.columns:
        test_df[col] = test_df[col].astype('category')

# Формируем временные признаки повторно
test_df['request_ts'] = pd.to_numeric(test_df['request_ts'], errors='coerce')
test_df['dt'] = pd.to_datetime(test_df['request_ts'], unit='s', errors='coerce')
test_df['hour'] = test_df['dt'].dt.hour
test_df['weekday'] = test_df['dt'].dt.weekday

# Формируем агрегированные признаки аналогично обучению
requests_count = test_df.groupby('user_id').size().rename('requests_count')

filtered_domains_test = test_df.dropna(subset=['user_id', 'referer_domain_top'])
domain_counts_test = pd.crosstab(filtered_domains_test['user_id'], filtered_domains_test['referer_domain_top'], dropna=False)
domain_freq_test = domain_counts_test.div(domain_counts_test.sum(axis=1), axis=0).fillna(0)
domain_freq_test.columns = [f'domain_freq_{col}' for col in domain_freq_test.columns]

filtered_browsers_test = test_df.dropna(subset=['user_id', 'browser_top'])
browser_counts_test = pd.crosstab(filtered_browsers_test['user_id'], filtered_browsers_test['browser_top'], dropna=False)
browser_freq_test = browser_counts_test.div(browser_counts_test.sum(axis=1), axis=0).fillna(0)
browser_freq_test.columns = [f'browser_freq_{col}' for col in browser_freq_test.columns]

filtered_oses_test = test_df.dropna(subset=['user_id', 'os_top'])
os_counts_test = pd.crosstab(filtered_oses_test['user_id'], filtered_oses_test['os_top'], dropna=False)
os_freq_test = os_counts_test.div(os_counts_test.sum(axis=1), axis=0).fillna(0)
os_freq_test.columns = [f'os_freq_{col}' for col in os_freq_test.columns]

component_cols_test = [f'component{i}' for i in range(10) if f'component{i}' in test_df.columns]
comp_agg_test = test_df.groupby('user_id')[component_cols_test].agg(['mean','median','max','min','std'])
comp_agg_test.columns = ['_'.join(col) for col in comp_agg_test.columns]

time_agg_test = test_df.groupby('user_id')['request_ts'].agg(['mean','min','max'])
time_agg_test['range'] = time_agg_test['max'] - time_agg_test['min']

hour_counts_test = pd.crosstab(test_df['user_id'], test_df['hour'])
hour_freq_test = hour_counts_test.div(hour_counts_test.sum(axis=1), axis=0).fillna(0)
hour_freq_test.columns = [f'hour_freq_{h}' for h in hour_freq_test.columns]

weekday_counts_test = pd.crosstab(test_df['user_id'], test_df['weekday'])
weekday_freq_test = weekday_counts_test.div(weekday_counts_test.sum(axis=1), axis=0).fillna(0)
weekday_freq_test.columns = [f'weekday_freq_{d}' for d in weekday_freq_test.columns]

# Объединяем признаки в итоговый датафрейм теста
test_features = pd.concat([
    requests_count,
    domain_freq_test,
    browser_freq_test,
    os_freq_test,
    comp_agg_test,
    time_agg_test,
    hour_freq_test,
    weekday_freq_test
], axis=1).fillna(0)

# Некоторые пользователи могут отсутствовать в тестовых признаках, добавляем для них строки с нулями
missing_users = set(test_users_df['user_id']) - set(test_features.index)
empty_features = pd.DataFrame(0, index=list(missing_users), columns=test_features.columns)
full_test_features = pd.concat([test_features, empty_features])

# Подтверждение того, что у признаков теста и обучения одинаковые колонки
train_feature_cols = train_features.drop(columns=['target']).columns
for col in train_feature_cols:
    if col not in full_test_features.columns:
        full_test_features[col] = 0

# Удаляем лишние колонки
extra_cols = [col for col in full_test_features.columns if col not in train_feature_cols]
if extra_cols:
    full_test_features.drop(columns=extra_cols, inplace=True)

# Переставляем колонки в том же порядке, как в обучении
full_test_features = full_test_features[train_feature_cols]

# Упорядочиваем строки
full_test_features = full_test_features.reindex(test_users_df['user_id'])

# Загружаем сохраненную модель
model = joblib.load(model_file)

# Делаем предсказания пола (0 или 1) для пользователей
pred_proba = model.predict(full_test_features, num_iteration=model.best_iteration)
pred_labels = (pred_proba > 0.5).astype(int)

# Формируем итоговый датафрейм с user_id и предсказанием пола
submission = pd.DataFrame({
    'user_id': test_users_df['user_id'],
    'target': pred_labels
})

# Сохраняем результат в CSV файл
submission.to_csv(submission_file, index=False, sep=';')

print(f"Файл '{submission_file}' успешно сохранён. Пример результатов:")
print(submission.head())
