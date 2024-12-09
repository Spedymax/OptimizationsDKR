import numpy as np
from scipy.optimize import minimize

def f1(x):
    return 2*x[0]**2 - x[0]*x[1] - x[1]**2 - 2.24

def f2(x):
    return x[0]**2 + x[1]**2 - x[0] - 1.88

def F(x):
    return f1(x)**2 + f2(x)**2

# Пошук мінімуму з різних початкових точок
result1 = minimize(F, [1.5, 1.2], method='Nelder-Mead')
result2 = minimize(F, [1.5, -1.2], method='Nelder-Mead')
result3 = minimize(F, [-1.5, -1.2], method='Nelder-Mead')

print("Мінімум 1:", result1.x, "зі значенням:", result1.fun)
print("Мінімум 2:", result2.x, "зі значенням:", result2.fun)
print("Мінімум 3:", result3.x, "зі значенням:", result3.fun)

# Побудуємо оновлений контурний графік
import matplotlib.pyplot as plt

x1 = np.linspace(-2, 3, 200)
x2 = np.linspace(-2, 3, 200)
X1, X2 = np.meshgrid(x1, x2)
Z = np.zeros_like(X1)

for i in range(len(x1)):
    for j in range(len(x2)):
        Z[i,j] = F([X1[i,j], X2[i,j]])

plt.figure(figsize=(12, 8))
plt.contour(X1, X2, Z, levels=np.logspace(-2, 2, 20))
plt.colorbar(label='F(x₁,x₂)')

# Позначаємо знайдені мінімуми
plt.plot(result1.x[0], result1.x[1], 'r*', markersize=15, label='Мінімум 1')
plt.plot(result2.x[0], result2.x[1], 'r*', markersize=15, label='Мінімум 2')
plt.plot(result3.x[0], result3.x[1], 'r*', markersize=15, label='Мінімум 3')

# Додаємо нові початкові точки
A0 = [-1.5, -1.5]
A1 = [3.0, 1.5]
plt.plot(A0[0], A0[1], 'go', markersize=10, label='A0(-1.5, -1.5)')
plt.plot(A1[0], A1[1], 'bo', markersize=10, label='A1(3.0, 1.5)')

plt.xlabel('x₁')
plt.ylabel('x₂')
plt.title('Контурний графік цільової функції')
plt.grid(True)
plt.legend()
plt.show()