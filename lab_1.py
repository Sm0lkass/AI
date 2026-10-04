import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# ==========================================
# 1. Загрузка данных и первичный осмотр
# ==========================================
# Замени 'titanic.csv' на имя твоего скачанного датасета с Kaggle
df = pd.read_csv(r'C:\Users\k513e\Downloads\titanic.csv')

print("--- Первые 5 строк датасета ---")
print(df.head())

print("\n--- Информация о датасете ---")
df.info()

# ==========================================
# 2. Подсчет пропущенных значений
# ==========================================
print("\n--- Пропуски до обработки ---")
print(df.isnull().sum())

# ==========================================
# 3. Заполнение пропущенных значений
# ==========================================
# Пример: Заполнение числовых признаков медианой/средним
if 'Age' in df.columns:
    age_median = df['Age'].median()
    df['Age'] = df['Age'].fillna(age_median)

if 'Fare' in df.columns:
    fare_mean = df['Fare'].mean()
    df['Fare'] = df['Fare'].fillna(fare_mean)

# Пример: Заполнение категориальных признаков модой
if 'Embarked' in df.columns:
    embarked_mode = df['Embarked'].mode()[0]
    df['Embarked'] = df['Embarked'].fillna(embarked_mode)

print("\n--- Пропуски после обработки (должно быть 0) ---")
print(df.isnull().sum())

# ==========================================
# 4. Нормализация данных
# ==========================================
# Масштабируем численные признаки (например, Age и Fare)
scaler = MinMaxScaler()  # Можно использовать StandardScaler()

numeric_cols = ['Age', 'Fare']
# Проверяем, есть ли такие колонки в нашем DataFrame
valid_num_cols = [col for col in numeric_cols if col in df.columns]

if valid_num_cols:
    df[valid_num_cols] = scaler.fit_transform(df[valid_num_cols])

print("\n--- Данные после нормализации ---")
print(df[valid_num_cols].head())

# ==========================================
# 5. Преобразование категориальных данных (One-Hot Encoding)
# ==========================================
# Кодируем категориальные переменные с флагом drop_first=True для избежания мультиколлинеарности
cat_cols = ['Sex', 'Embarked']
valid_cat_cols = [col for col in cat_cols if col in df.columns]

if valid_cat_cols:
    df = pd.get_dummies(df, columns=valid_cat_cols, drop_first=True)

print("\n--- Итоговый датасет после One-Hot Encoding ---")
print(df.head())

# ==========================================
# 6. Сохранение обработанных данных
# ==========================================
df.to_csv('processed_titanic.csv', index=False)
print("\nОбработанный датасет успешно сохранен в 'processed_titanic.csv'!")