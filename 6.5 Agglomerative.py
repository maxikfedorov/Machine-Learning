import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering

# Загрузка данных
data = pd.read_csv('planets.csv')

# Выбор признаков для кластеризации
features = data[['distance', 'stellar_magnitude', 'mass_multiplier', 'radius_multiplier', 'orbital_radius']]

# Обработка NaN и бесконечных значений
features.replace([np.inf, -np.inf], np.nan, inplace=True)  # Замена бесконечных значений на NaN
features.fillna(features.mean(), inplace=True)  # Замена NaN на средние значения

# Нормализация данных
features = (features - features.mean()) / features.std()

# Применение агломеративной кластеризации с методом Уорда
n_clusters = 3  # Количество кластеров
clustering = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
labels = clustering.fit_predict(features)

# Добавление меток кластеров в исходные данные
data['cluster'] = labels

# Вывод результатов кластеризации
print("Кластеризация планет:")
print(data[['name', 'cluster']].head())  # Первые несколько строк
print("...")
print(data[['name', 'cluster']].tail())  # Последние несколько строк

# Подсчет количества элементов в каждом кластере
cluster_counts = data['cluster'].value_counts().sort_index()
print("\nКоличество элементов в каждом кластере:")
for cluster_num, count in cluster_counts.items():
    print(f"Кластер {cluster_num}: {count} элементов")

# Визуализация результатов кластеризации
plt.figure(figsize=(10, 6))
colors = ['red', 'blue', 'green']

for cluster_num in range(n_clusters):
    cluster_data = features.values[labels == cluster_num]
    plt.scatter(cluster_data[:, 0], cluster_data[:, 1], s=50, c=colors[cluster_num], label=f'Кластер {cluster_num}')

plt.xlabel('Distance (нормализовано)')
plt.ylabel('Stellar Magnitude (нормализовано)')
plt.title('Агломеративная кластеризация методом Уорда')
plt.legend()
plt.show()
