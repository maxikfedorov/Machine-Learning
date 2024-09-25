import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt

# Загрузка данных из CSV файла
data = pd.read_csv('planets.csv')

# Отбор числовых столбцов для кластеризации
features = ['distance', 'stellar_magnitude', 'mass_multiplier', 'radius_multiplier', 'orbital_radius', 'orbital_period', 'eccentricity']

# Удаление строк с пропущенными значениями (или можно использовать заполнение)
data = data.dropna()

# Масштабирование данных
scaler = StandardScaler()
scaled_features = scaler.fit_transform(data[features])

# Определение количества кластеров с помощью метода локтя
inertia = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(scaled_features)
    inertia.append(kmeans.inertia_)

# Построение графика метода локтя
plt.figure(figsize=(8, 5))
plt.plot(range(1, 11), inertia, marker='o')
plt.title('Метод локтя')
plt.xlabel('Количество кластеров')
plt.ylabel('Inertia')
plt.show()

# Обучение модели K-средних с оптимальным количеством кластеров
optimal_k = 4
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
data['Cluster'] = kmeans.fit_predict(scaled_features)

# Вывод результатов кластеризации
print(data[['name', 'Cluster']].head())

# Визуализация результатов (на примере первых двух признаков)
plt.figure(figsize=(8, 5))
plt.scatter(scaled_features[:, 0], scaled_features[:, 1], c=data['Cluster'], cmap='viridis')
plt.title('Кластеризация методом K-средних')
plt.xlabel(features[0])
plt.ylabel(features[1])
plt.show()
