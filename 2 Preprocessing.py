import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import Binarizer, StandardScaler, MinMaxScaler, Normalizer

# Загрузка данных из CSV файла
data = pd.read_csv('planets.csv')

# Поиск и удаление дубликатов
initial_shape = data.shape
data.drop_duplicates(inplace=True)
print(f"Удаление дубликатов: {initial_shape[0] - data.shape[0]} дубликатов удалено")

# Выбор числовых столбцов для обработки
numeric_columns = ['distance', 'stellar_magnitude', 'mass_multiplier', 'radius_multiplier', 'orbital_radius', 'orbital_period', 'eccentricity']

# Иммутация пропущенных значений
imputer = SimpleImputer(strategy='mean')
data[numeric_columns] = imputer.fit_transform(data[numeric_columns])
print("Иммутация пропущенных значений выполнена.")

# Бинаризация данных (например, бинаризация расстояния)
binarizer = Binarizer(threshold=300)  # Пороговое значение для бинаризации
data['distance_binarized'] = binarizer.fit_transform(data[['distance']])
print("Бинаризация столбца 'distance' выполнена.")

# Исключение среднего и масштабирование (StandardScaler)
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data[numeric_columns])
print("Исключение среднего и масштабирование выполнены.")

# Масштабирование данных в диапазон [0, 1] (MinMaxScaler)
min_max_scaler = MinMaxScaler()
scaled_data_minmax = min_max_scaler.fit_transform(data[numeric_columns])
print("Масштабирование в диапазон [0, 1] выполнено.")

# Нормализация данных (Normalizer)
normalizer = Normalizer()
normalized_data = normalizer.fit_transform(data[numeric_columns])
print("Нормализация данных выполнена.")

# Пример вывода первых нескольких строк обработанных данных
print("\nПример первых нескольких строк обработанных данных:")
print(data.head())

# Пример статистики после каждого этапа
print("\nСтатистика после каждого этапа:")
print("\nСтатистика после исключения среднего и масштабирования:")
print(pd.DataFrame(scaled_data, columns=numeric_columns).describe())

print("\nСтатистика после масштабирования в диапазон [0, 1]:")
print(pd.DataFrame(scaled_data_minmax, columns=numeric_columns).describe())

print("\nСтатистика после нормализации:")
print(pd.DataFrame(normalized_data, columns=numeric_columns).describe())




