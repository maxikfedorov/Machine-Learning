import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Загрузка данных из CSV файла
data = pd.read_csv('planets.csv')

# Просмотр первых строк данных
print(data.head())

# Проверка информации о данных
print(data.info())

# Проверка наличия пропущенных значений
print(data.isnull().sum())

# Анализ распределения открытий по годам
plt.figure(figsize=(10, 6))
sns.countplot(x='discovery_year', data=data)
plt.title('Распределение открытий по годам')
plt.xticks(rotation=45)
plt.show()

# Преобразуем год открытия в категориальную переменную (например, по десятилетиям)
data['decade'] = (data['discovery_year'] // 10) * 10

plt.figure(figsize=(10, 6))
sns.countplot(x='decade', data=data)
plt.title('Распределение открытий по десятилетиям')
plt.xticks(rotation=45)
plt.show()

# Анализ зависимости между массой и радиусом планеты
plt.figure(figsize=(10, 6))
sns.scatterplot(x='mass_multiplier', y='radius_multiplier', data=data)
plt.title('Зависимость массы от радиуса планеты')
plt.xlabel('Масса (в кратных массах Юпитера)')
plt.ylabel('Радиус (в кратных радиусах Юпитера)')
plt.show()

# Анализ зависимости между орбитальным радиусом и периодом
plt.figure(figsize=(10, 6))
sns.scatterplot(x='orbital_radius', y='orbital_period', data=data)
plt.title('Зависимость орбитального радиуса от периода')
plt.xlabel('Орбитальный радиус')
plt.ylabel('Орбитальный период')
plt.show()

# Проверка корреляции между числовыми переменными
numeric_data = data.select_dtypes(include=['float64', 'int64'])
correlation_matrix = numeric_data.corr()
print(correlation_matrix)

plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Корреляционная матрица')
plt.show()
