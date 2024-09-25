import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans

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

# Метод локтя для KMeans
sse = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=0)
    kmeans.fit(features_scaled)
    sse.append(kmeans.inertia_)

plt.figure(figsize=(10, 5))
plt.plot(range(1, 11), sse, marker='o')
plt.title('Метод локтя для KMeans')
plt.xlabel('Количество кластеров')
plt.ylabel('SSE')
plt.show()

# Информационные критерии для GMM
n_components_range = range(1, 11)
aic = []
bic = []

for n_components in n_components_range:
    gmm = GaussianMixture(n_components=n_components, random_state=0)
    gmm.fit(features_scaled)
    aic.append(gmm.aic(features_scaled))
    bic.append(gmm.bic(features_scaled))

plt.figure(figsize=(10, 5))
plt.plot(n_components_range, aic, label='AIC', marker='o')
plt.plot(n_components_range, bic, label='BIC', marker='o')
plt.title('Информационные критерии для GMM')
plt.xlabel('Количество компонентов')
plt.ylabel('Значение критерия')
plt.legend()
plt.show()
