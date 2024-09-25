import pandas as pd
import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# Загрузка данных
data = pd.read_csv('planets.csv')

# Выбор числовых столбцов для кластеризации
features = ['distance', 'stellar_magnitude', 'discovery_year', 'mass_multiplier',
            'radius_multiplier', 'orbital_radius', 'orbital_period', 'eccentricity']
X = data[features]

# Преобразование данных в числовой формат и обработка отсутствующих значений
X = X.apply(pd.to_numeric, errors='coerce')

# Заполнение отсутствующих значений средними значениями столбцов
X.fillna(X.mean(), inplace=True)

# Стандартизация данных
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Проверка на наличие бесконечных значений после стандартизации
if np.any(np.isinf(X_scaled)) or np.any(np.isnan(X_scaled)):
    raise ValueError("Стандартизированные данные содержат бесконечные или отсутствующие значения.")

# Выполнение иерархической кластеризации
Z = linkage(X_scaled, method='ward')

# Ограничение на количество кластеров (например, 5)
max_clusters = 8
clusters = fcluster(Z, max_clusters, criterion='maxclust')

# Добавление информации о кластерах в исходные данные
data['cluster'] = clusters

# Подсчет количества объектов в каждом кластере
cluster_counts = data['cluster'].value_counts().sort_index()

# Вывод количества объектов в каждом кластере
print("Количество объектов в каждом кластере:")
for cluster_num, count in cluster_counts.items():
    print(f"Кластер {cluster_num}: {count} объектов")

# Визуализация дендрограммы (можно ограничить количество отображаемых объектов)
plt.figure(figsize=(10, 7))
dendrogram(Z, truncate_mode='lastp', p=max_clusters, show_leaf_counts=True)
plt.title('Дендрограмма')
plt.xlabel('Кластер')
plt.ylabel('Расстояние')
plt.show()

# Сохранение данных с кластерами в новый CSV файл
data.to_csv('planets_with_clusters.csv', index=False)

print("Кластеризация завершена. Результаты сохранены в файл 'planets_with_clusters.csv'.")
