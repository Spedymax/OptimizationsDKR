import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def f1(x1, x2):
    return 2*x1**2 - x1*x2 - x2**2 - 2.24

def f2(x1, x2):
    return x1**2 + x2**2 - x1 - 1.88

def F(x1, x2):
    return f1(x1, x2)**2 + f2(x1, x2)**2

# Створюємо сітку точок з меншим кроком для кращої деталізації
x1 = np.linspace(-2, 3, 200)
x2 = np.linspace(-2, 3, 200)
X1, X2 = np.meshgrid(x1, x2)

# Обчислюємо значення цільової функції
Z = np.zeros_like(X1)
for i in range(len(x1)):
    for j in range(len(x2)):
        Z[i,j] = min(F(X1[i,j], X2[i,j]), 10)  # Обмежуємо максимальне значення

# Створюємо графік
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# Будуємо поверхню з покращеними параметрами
surface = ax.plot_surface(X1, X2, Z, cmap='viridis',
                         alpha=0.9,
                         rstride=1,
                         cstride=1,
                         linewidth=0,
                         antialiased=True)

# Додаємо кольорову шкалу
plt.colorbar(surface, ax=ax, shrink=0.5, aspect=5)

# Налаштовуємо осі та заголовок
ax.set_xlabel('x₁')
ax.set_ylabel('x₂')
ax.set_zlabel('F(x₁,x₂)')

# Встановлюємо обмеження осей
ax.set_zlim(0, 10)
ax.set_xlim(-2, 3)
ax.set_ylim(-2, 3)

# Встановлюємо кут огляду для кращого представлення
ax.view_init(elev=30, azim=45)

plt.show()

# Будуємо також контурний графік для кращого розуміння
plt.figure(figsize=(10, 8))
plt.contour(X1, X2, Z, levels=np.logspace(-2, 1, 20))
plt.colorbar(label='F(x₁,x₂)')
plt.xlabel('x₁')
plt.ylabel('x₂')
plt.grid(True)
plt.show()