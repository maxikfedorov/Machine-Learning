import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Загрузка данных
data = pd.read_csv('planets.csv')

# Просмотр первых нескольких строк данных
print("### Первые строки данных ###")
print(data.head())

# **Эмпирические распределения**
print("\n### Эмпирические распределения ###")
numeric_features = ['distance', 'stellar_magnitude', 'mass_multiplier', 'radius_multiplier', 
                    'orbital_radius', 'orbital_period', 'eccentricity']

for feature in numeric_features:
    plt.figure(figsize=(10, 4))
    if feature in ['orbital_radius', 'orbital_period']:
        sns.histplot(data[feature], kde=True, log_scale=True)  # Использование логарифмической шкалы
        plt.title(f'Распределение {feature} (логарифмическая шкала)')
    else:
        sns.histplot(data[feature], kde=True)
        plt.title(f'Распределение {feature}')
    plt.xlabel(feature)
    plt.ylabel('Частота')
    plt.show()

# **Выявление трендов**
print("\n### Выявление трендов ###")
trend_data = data.groupby('discovery_year')['mass_multiplier'].mean().reset_index()
plt.figure(figsize=(10, 6))
sns.lineplot(x='discovery_year', y='mass_multiplier', data=trend_data)
plt.title('Средняя масса планет по годам открытия')
plt.xlabel('Год открытия')
plt.ylabel('Средняя масса (в единицах массы Юпитера)')
plt.grid(True)
plt.show()

# **Определение статистических характеристик**
print("\n### Определение статистических характеристик ###")
stats_summary = data[numeric_features].describe(percentiles=[.25, .5, .75]).T
stats_summary['skew'] = data[numeric_features].skew()
stats_summary['kurtosis'] = data[numeric_features].kurtosis()

print("Статистические характеристики:")
print(stats_summary)
