
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Circle

# Задание 1 (1 балл)
# Сгенерируйте массив нормально распределенных значений размерности 2 из 100 точек
# (выберите среднее значение μ и среднее квадратическое отклонение σ по своему выбору).
# Проверьте правило трех сигм: нарисуйте окружность с центром в точке μ с таким радиусом,
# чтобы на нее приходилось 0,99 всех точек, а также окружность радиусом 3 сигмы.
# Выделите точку μ отдельным цветом.

mu_x, mu_y = 0, 0
sigma = 1
n_points = 100


np.random.seed(42)
points = np.random.multivariate_normal([mu_x, mu_y], [[sigma, 0], [0, sigma]], n_points)

x_coords = points[:, 0]
y_coords = points[:, 1]

radius_99 = 2.58 * sigma
radius_3sigma = 3 * sigma

fig, ax = plt.subplots(figsize=(8, 8))

ax.scatter(x_coords, y_coords, c='blue', alpha=0.6, label='Точки данных')

ax.scatter(mu_x, mu_y, c='red', s=100, label=f'Точка μ ({mu_x}, {mu_y})', zorder=5)

circle_99 = Circle((mu_x, mu_y), radius_99, color='green', fill=False, linestyle='--', linewidth=2, label=f'99% точек (r={radius_99:.2f})')
ax.add_patch(circle_99)

circle_3sigma = Circle((mu_x, mu_y), radius_3sigma, color='orange', fill=False, linestyle='-.', linewidth=2, label=f'3σ (r={radius_3sigma})')
ax.add_patch(circle_3sigma)


ax.set_xlim(-6, 6)
ax.set_ylim(-6, 6)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_title('Нормальное распределение 2D и правило 3-х сигм')
ax.legend()
ax.grid(True)

plt.axis('equal')
plt.show()


# Задание 2 (1 балл)
# Используйте вспомогательный график, чтобы нарисовать гистограммы с 10 сегментами для каждого измерения
# и построить график плотности вдоль гистограммы для данных из первого задания.

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))


ax1.hist(x_coords, bins=10, density=True, alpha=0.6, color='skyblue', edgecolor='black', label='Гистограмма X')

x_range = np.linspace(x_coords.min(), x_coords.max(), 100)
ax1.plot(x_range,
         (1/(sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x_range - mu_x)/sigma)**2),
         color='red', label='Теоретическая плотность X')
ax1.set_xlabel('X')
ax1.set_ylabel('Плотность')
ax1.set_title('Гистограмма и плотность для X')
ax1.legend()
ax1.grid(True)


ax2.hist(y_coords, bins=10, density=True, alpha=0.6, color='lightcoral', edgecolor='black', label='Гистограмма Y')

y_range = np.linspace(y_coords.min(), y_coords.max(), 100)
ax2.plot(y_range,
         (1/(sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((y_range - mu_y)/sigma)**2),
         color='red', label='Теоретическая плотность Y')
ax2.set_xlabel('Y')
ax2.set_ylabel('Плотность')
ax2.set_title('Гистограмма и плотность для Y')
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.show()