import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage

# Загрузка данных
data = pd.read_csv('planets.csv')

# Выбор признаков для кластеризации
features = data[['distance', 'stellar_magnitude', 'mass_multiplier', 'radius_multiplier', 'orbital_radius']]

# Обработка NaN и бесконечных значений
features.replace([np.inf, -np.inf], np.nan, inplace=True)
features.fillna(features.mean(), inplace=True)

# Нормализация данных
features = (features - features.mean()) / features.std()

# Построение модели linkage для дендрограммы
linked = linkage(features, method='ward')

# Построение дендрограммы с ограничением количества кластеров
plt.figure(figsize=(10, 7))
dendrogram(linked,
           orientation='top',
           labels=data['name'].values,
           distance_sort='descending',
           show_leaf_counts=False,
           truncate_mode='lastp',  # Ограничивает количество кластеров
           p=100)  # Установите желаемое количество кластеров
plt.title('Дендрограмма планет')
plt.xlabel('Планеты')
plt.ylabel('Расстояние')
plt.show()
