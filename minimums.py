import numpy as np
from scipy.optimize import minimize

def f1(x):
    return 2*x[0]**2 - x[0]*x[1] - x[1]**2 - 2.24

def f2(x):
    return x[0]**2 + x[1]**2 - x[0] - 1.88

def F(x):
    return f1(x)**2 + f2(x)**2

starting_points = [
    [-1, 0],
    [1.5, 1],
    [1.5, -1],
    [-1, -1],
    [0, 0],
    [2, 0],
]

minima = []
for start in starting_points:
    result = minimize(F, start, method='Nelder-Mead', tol=1e-6)
    if result.success:
        is_new = True
        for existing_min in minima:
            if np.allclose(existing_min[:2], result.x, rtol=1e-3, atol=1e-3):
                is_new = False
                break
        if is_new and result.fun < 0.1:
            minima.append(np.append(result.x, result.fun))

print("Найденные минимумы (x1, x2, значение функции):")
for i, min_point in enumerate(minima, 1):
    print(f"Минимум {i}: ({min_point[0]:.4f}, {min_point[1]:.4f}), F = {min_point[2]:.6f}")

import matplotlib.pyplot as plt

x1 = np.linspace(-2, 3, 200)
x2 = np.linspace(-2, 3, 200)
X1, X2 = np.meshgrid(x1, x2)
Z = np.array([[F([x1_val, x2_val]) for x1_val, x2_val in zip(x1_row, x2_row)]
              for x1_row, x2_row in zip(X1, X2)])

plt.figure(figsize=(12, 8))
plt.contour(X1, X2, Z, levels=np.logspace(-2, 2, 20))
plt.colorbar(label='F(x₁,x₂)')

for i, min_point in enumerate(minima, 1):
    plt.plot(min_point[0], min_point[1], 'r*', markersize=15, label=f'Минимум {i}')

A0 = [-1.5, -1.5]
A1 = [3.0, 1.5]
plt.plot(A0[0], A0[1], 'go', markersize=10, label='A0(-1.5, -1.5)')
plt.plot(A1[0], A1[1], 'bo', markersize=10, label='A1(3.0, 1.5)')

plt.xlabel('x₁')
plt.ylabel('x₂')
plt.title('Контурний графік функції')
plt.grid(True)
plt.legend()
plt.show()