import pandas as pd
import numpy as np
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import StandardScaler
import random
import matplotlib.pyplot as plt

# Загрузка набора данных
data = pd.read_csv('planets.csv')

# Выбор релевантных признаков для кластеризации
features = data[['distance', 'stellar_magnitude', 'mass_multiplier', 'radius_multiplier', 'orbital_radius']]

# Обработка отсутствующих значений: заполнение средним значением каждого столбца
features.fillna(features.mean(), inplace=True)

# Стандартизация признаков
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

def k_medoids(X, k, max_iter=100):
    # Случайная инициализация медоидов
    m = X.shape[0]
    medoid_indices = random.sample(range(m), k)
    medoids = X[medoid_indices]
    
    for _ in range(max_iter):
        # Вычисление расстояний между точками и медоидами
        distances = pairwise_distances(X, medoids, metric='euclidean')
        
        # Назначение каждой точки ближайшему медоиду
        labels = np.argmin(distances, axis=1)
        
        # Обновление медоидов
        new_medoids = np.copy(medoids)
        
        for i in range(k):
            cluster_points = X[labels == i]
            if len(cluster_points) == 0:
                continue
            
            # Расчет стоимости для каждой точки в кластере, чтобы стать медоидом
            intra_cluster_distances = pairwise_distances(cluster_points, cluster_points, metric='euclidean')
            costs = np.sum(intra_cluster_distances, axis=1)
            
            # Выбор точки с минимальной стоимостью в качестве нового медоида
            new_medoids[i] = cluster_points[np.argmin(costs)]
        
        # Проверка на сходимость (если медоиды не изменились)
        if np.all(medoids == new_medoids):
            break
        
        medoids = new_medoids
    
    return labels, medoids

# Применение алгоритма k-медоидов
k = 3  # Количество кластеров
labels, medoids = k_medoids(scaled_features, k)

# Добавление меток кластеров в исходные данные
data['Cluster'] = labels

# Визуализация результатов кластеризации
plt.figure(figsize=(10, 6))
plt.scatter(scaled_features[:, 0], scaled_features[:, 1], c=labels, cmap='viridis', marker='o')
plt.scatter(medoids[:, 0], medoids[:, 1], c='red', marker='x', s=100)
plt.title('K-Medoids Clustering')
plt.xlabel('Distance (standardized)')
plt.ylabel('Stellar Magnitude (standardized)')
plt.show()
