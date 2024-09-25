import pandas as pd
import matplotlib.pyplot as plt

# Загрузка данных из файла CSV
data = pd.read_csv('planets.csv')

# Вывод первых нескольких строк для ознакомления с данными
print("Исходные данные:")
print(data.head())

# Проверка наличия выбросов с помощью диаграммы размаха (boxplot)
numeric_columns = data.select_dtypes(include=['float64', 'int64']).columns

# Построение диаграмм размаха для всех числовых столбцов
for column in numeric_columns:
    plt.figure(figsize=(10, 5))
    plt.boxplot(data[column].dropna(), vert=False)
    plt.title(f'Boxplot for {column}')
    plt.xlabel(column)
    plt.show()

# Обработка выбросов с использованием метода межквартильного размаха (IQR)
def remove_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

# Применение функции для удаления выбросов в каждом числовом столбце
for column in numeric_columns:
    data = remove_outliers(data, column)

# Проверка данных после удаления выбросов
print("\nДанные после удаления выбросов:")
print(data.head())

# Сохранение обработанных данных в новый CSV файл
data.to_csv('data/planets_no_outliers.csv', index=False)
