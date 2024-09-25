import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

# Загрузка данных
data = pd.read_csv('planets.csv')

# Выбор признаков для кластеризации
features = data[['distance', 'stellar_magnitude', 'mass_multiplier', 'radius_multiplier', 'orbital_radius']]

# Обработка NaN и бесконечных значений
features.replace([np.inf, -np.inf], np.nan, inplace=True)  # Замена бесконечных значений на NaN
features.fillna(features.mean(), inplace=True)  # Замена NaN на средние значения

# Нормализация данных
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# Применение DBSCAN
dbscan = DBSCAN(eps=0.5, min_samples=5)
labels = dbscan.fit_predict(features_scaled)

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
unique_labels = set(labels)
colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]

for k, col in zip(unique_labels, colors):
    if k == -1:
        # Черный цвет для шума.
        col = [0, 0, 0, 1]

    class_member_mask = (labels == k)
    xy = features_scaled[class_member_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col), markeredgecolor='k', markersize=6)

plt.title('DBSCAN: Кластеризация планет')
plt.xlabel('Distance (нормализовано)')
plt.ylabel('Stellar Magnitude (нормализовано)')
plt.show()
