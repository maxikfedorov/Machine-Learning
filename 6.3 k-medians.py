import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import pairwise_distances_argmin

# Загрузка данных
data = pd.read_csv('planets.csv')

# Выбор признаков для кластеризации
features = data[['distance', 'stellar_magnitude', 'mass_multiplier', 'radius_multiplier', 'orbital_radius']]

# Обработка NaN и бесконечных значений
features.replace([np.inf, -np.inf], np.nan, inplace=True)  # Замена бесконечных значений на NaN
features.fillna(features.mean(), inplace=True)  # Замена NaN на средние значения

# Нормализация данных
features = (features - features.mean()) / features.std()

def k_medians(X, n_clusters, max_iter=100):
    # Случайный выбор начальных центроидов
    rng = np.random.RandomState(42)
    i = rng.permutation(X.shape[0])[:n_clusters]
    medians = X[i]

    for _ in range(max_iter):
        # Назначение каждой точки к ближайшему центроиду
        labels = pairwise_distances_argmin(X, medians)
        
        # Вычисление новых медиан для каждого кластера
        new_medians = np.array([np.median(X[labels == i], axis=0) for i in range(n_clusters)])
        
        # Проверка на сходимость
        if np.all(medians == new_medians):
            break
        
        medians = new_medians
    
    return medians, labels

# Применение алгоритма k-медиан
n_clusters = 3  # Количество кластеров
medians, labels = k_medians(features.values, n_clusters)

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

plt.scatter(medians[:, 0], medians[:, 1], s=200, c='yellow', marker='X', label='Медианы')
plt.xlabel('Distance (нормализовано)')
plt.ylabel('Stellar Magnitude (нормализовано)')
plt.title('Кластеризация планет методом k-медиан')
plt.legend()
plt.show()
