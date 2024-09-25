import pandas as pd
from sklearn.impute import SimpleImputer

# Загрузка данных из файла CSV
data = pd.read_csv('planets.csv')

# Вывод первых нескольких строк для ознакомления с данными
print("Исходные данные:")
print(data.head())

# Проверка наличия пропусков в данных
missing_data = data.isnull().sum()
print("\nКоличество пропусков в каждом столбце:")
print(missing_data)

# Создание объекта SimpleImputer для заполнения пропусков средним значением
imputer = SimpleImputer(strategy='mean')

# Заполнение пропусков в числовых столбцах
numeric_columns = data.select_dtypes(include=['float64', 'int64']).columns
data[numeric_columns] = imputer.fit_transform(data[numeric_columns])

# Проверка данных после заполнения пропусков
print("\nДанные после заполнения пропусков:")
print(data.head())

# Сохранение обработанных данных в новый CSV файл
data.to_csv('data/planets_imputed.csv', index=False)
