import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Загрузка данных из CSV файла
data = pd.read_csv('planets.csv')

# Создание графика распределения расстояний до планет
plt.figure(figsize=(10, 6))
sns.histplot(data['distance'], bins=30, kde=True)
plt.title('Распределение расстояний до планет')
plt.xlabel('Расстояние (световые годы)')
plt.ylabel('Количество планет')
plt.show()

# График зависимости массы от радиуса планет
plt.figure(figsize=(10, 6))
sns.scatterplot(x='radius_multiplier', y='mass_multiplier', hue='planet_type', data=data)
plt.title('Зависимость массы от радиуса планет')
plt.xlabel('Радиус (относительно Юпитера)')
plt.ylabel('Масса (относительно Юпитера)')
plt.legend(title='Тип планеты')
plt.show()

# Круговая диаграмма по методам обнаружения
detection_counts = data['detection_method'].value_counts()
colors = plt.cm.Paired(range(len(detection_counts)))
explode = [0.1] * len(detection_counts)  # Выделение всех сегментов

def autopct_format(pct):
    return ('%1.1f%%' % pct) if pct >= 2 else ''

plt.figure(figsize=(8, 8))
wedges, texts, autotexts = plt.pie(
    detection_counts,
    autopct=autopct_format,
    startangle=140,
    colors=colors,
    explode=explode,
    wedgeprops=dict(edgecolor='w', linewidth=1)
)

# Настройка автотекста
for autotext in autotexts:
    autotext.set_color('white')
    autotext.set_fontsize(10)

# Формирование подписей для легенды с процентами
legend_labels = [f'{label}: {pct:.1f}%' for label, pct in zip(detection_counts.index, 100 * detection_counts / detection_counts.sum())]

# Расположение легенды
plt.legend(wedges, legend_labels, bbox_to_anchor=(1, 0.5), loc='center left', title='Метод обнаружения')
plt.title('Методы обнаружения планет')
plt.axis('equal')
plt.tight_layout()  # Автоматическая подгонка элементов
plt.show()
