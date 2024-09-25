import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Загрузка данных
data = pd.read_csv('planets.csv')

# Выбор признаков для кластеризации
features = data[['distance', 'stellar_magnitude', 'mass_multiplier', 'radius_multiplier', 'orbital_radius']]

# Обработка NaN и бесконечных значений
features.replace([np.inf, -np.inf], np.nan, inplace=True)
features.fillna(features.mean(), inplace=True)

# Нормализация данных
features = (features - features.mean()) / features.std()

def k_means(X, n_clusters, max_iter=100):
    # Случайный выбор начальных центроидов
    rng = np.random.RandomState(42)
    centroids = X[rng.choice(X.shape[0], n_clusters, replace=False)]
    
    for _ in range(max_iter):
        # Назначение каждой точки к ближайшему центроиду
        distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)
        labels = np.argmin(distances, axis=1)
        
        # Вычисление новых центроидов
        new_centroids = np.array([X[labels == i].mean(axis=0) for i in range(n_clusters)])
        
        # Проверка на сходимость
        if np.all(centroids == new_centroids):
            break
        
        centroids = new_centroids
    
    return centroids, labels

# Применение алгоритма K-средних
n_clusters = 3
centroids, labels = k_means(features.values, n_clusters)

# Добавление меток кластеров в исходные данные
data['cluster'] = labels

# Визуализация результатов кластеризации
plt.figure(figsize=(10, 6))
colors = ['red', 'blue', 'green']

for cluster_num in range(n_clusters):
    cluster_data = features.values[labels == cluster_num]
    plt.scatter(cluster_data[:, 0], cluster_data[:, 1], s=50, c=colors[cluster_num], label=f'Кластер {cluster_num}')

plt.scatter(centroids[:, 0], centroids[:, 1], s=200, c='yellow', marker='X', label='Центроиды')
plt.xlabel('Distance (нормализовано)')
plt.ylabel('Stellar Magnitude (нормализовано)')
plt.title('Кластеризация методом K-средних')
plt.legend()
plt.show()
