import numpy as np
import matplotlib.pyplot as plt

def f1(x1, x2):
    return 2*x1**2 - x1*x2 - x2**2 - 2.24

def f2(x1, x2):
    return x1**2 + x2**2 - x1 - 1.88

def F(x1, x2):
    return f1(x1, x2)**2 + f2(x1, x2)**2

# Створюємо сітку точок
x1 = np.linspace(-3, 3, 100)
x2 = np.linspace(-3, 3, 100)
X1, X2 = np.meshgrid(x1, x2)

# Обчислюємо значення цільової функції
Z = F(X1, X2)

# Створюємо 3D графік
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Будуємо поверхню
surf = ax.plot_surface(X1, X2, Z, cmap='viridis')

ax.set_xlabel('x₁')
ax.set_ylabel('x₂')
ax.set_zlabel('F(x₁,x₂)')
plt.title('Цільова функція')

# Додаємо кольорову шкалу
plt.colorbar(surf)

plt.show()