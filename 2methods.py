import numpy as np
import matplotlib.pyplot as plt


def f1(x1, x2):
    return 2 * x1 ** 2 - x1 * x2 - x2 ** 2 - 2.24


def f2(x1, x2):
    return x1 ** 2 + x2 ** 2 - x1 - 1.88


def F(x1, x2):
    return f1(x1, x2) ** 2 + f2(x1, x2) ** 2


class CoordinateDescent:
    def __init__(self):
        self.eps = 1e-3
        self.max_iterations = 100
        self.func_calls = 0
        self.all_points = []

    def golden_section(self, f, a, b, tol=1e-5):
        gold = (1 + np.sqrt(5)) / 2
        while abs(b - a) > tol:
            c = b - (b - a) / gold
            d = a + (b - a) / gold
            if f(c) < f(d):
                b = d
            else:
                a = c
            self.func_calls += 2
        return (a + b) / 2

    def optimize(self, x0):
        x = np.array(x0, dtype=float)
        self.all_points = [x.copy()]  # Початкова точка

        for iter in range(self.max_iterations):
            x_old = x.copy()

            for i in range(2):
                def f(alpha):
                    x_tmp = x.copy()
                    x_tmp[i] = alpha
                    return F(x_tmp[0], x_tmp[1])

                self.all_points.append(x.copy())

                x[i] = self.golden_section(f, x[i] - 2, x[i] + 2)

                self.all_points.append(x.copy())

            if np.all(np.abs(x - x_old) < self.eps):
                break

        return {
            'minimum': x,
            'path': np.array(self.all_points),
            'value': F(x[0], x[1]),
            'func_calls': self.func_calls
        }


class GradientDescent:
    def __init__(self):
        self.eps = 1e-3
        self.h = 0.01
        self.max_iterations = 100
        self.func_calls = 0
        self.all_points = []

    def gradient(self, x):
        grad = np.zeros(2)
        for i in range(2):
            x_plus = x.copy()
            x_minus = x.copy()
            x_plus[i] += self.h
            x_minus[i] -= self.h
            grad[i] = (F(x_plus[0], x_plus[1]) - F(x_minus[0], x_minus[1])) / (2 * self.h)
            self.func_calls += 2
        return grad

    def optimize(self, x0):
        x = np.array(x0, dtype=float)
        self.all_points = [x.copy()]

        for iter in range(self.max_iterations):
            grad = self.gradient(x)
            if np.linalg.norm(grad) < self.eps:
                break

            direction = -grad / np.linalg.norm(grad)

            alpha = 0.1
            while F(x[0] + alpha * direction[0], x[1] + alpha * direction[1]) >= F(x[0], x[1]):
                alpha *= 0.5
                if alpha < self.eps:
                    break

            x_new = x + alpha * direction

            self.all_points.append(x_new.copy())

            if alpha * np.linalg.norm(direction) < self.eps:
                break

            x = x_new

        return {
            'minimum': x,
            'path': np.array(self.all_points),
            'value': F(x[0], x[1]),
            'func_calls': self.func_calls
        }


def plot_optimization(method_name, A0, A1, result_A0, result_A1):
    x1 = np.linspace(-2, 3, 200)
    x2 = np.linspace(-2, 3, 200)
    X1, X2 = np.meshgrid(x1, x2)
    Z = np.array([[F(x1_val, x2_val) for x1_val, x2_val in zip(x1_row, x2_row)]
                  for x1_row, x2_row in zip(X1, X2)])

    plt.figure(figsize=(12, 8))
    plt.contour(X1, X2, Z, levels=np.logspace(-2, 2, 20))
    plt.colorbar(label='F(x₁,x₂)')

    path_A0 = result_A0['path']
    path_A1 = result_A1['path']

    print(f"Кількість точок у траєкторії A0: {len(path_A0)}")
    print(f"Кількість точок у траєкторії A1: {len(path_A1)}")

    plt.plot(path_A0[:, 0], path_A0[:, 1], 'g.-', linewidth=1.5,
             markersize=3, label='Траєкторія з A0')

    plt.plot(path_A1[:, 0], path_A1[:, 1], 'b.-', linewidth=1.5,
             markersize=3, label='Траєкторія з A1')

    plt.plot(A0[0], A0[1], 'go', markersize=10, label='A0')
    plt.plot(A1[0], A1[1], 'bo', markersize=10, label='A1')

    plt.xlabel('x₁')
    plt.ylabel('x₂')
    plt.title(f'Траєкторії методу {method_name}')
    plt.grid(True)
    plt.legend()
    plt.show()

    print(f"\nРезультати {method_name}:")
    print(f"A0 -> мінімум: {result_A0['minimum']}, значення: {result_A0['value']:.6f}")
    print(f"A1 -> мінімум: {result_A1['minimum']}, значення: {result_A1['value']:.6f}")
    print(f"Кількість обчислень функції для A0: {result_A0['func_calls']}")
    print(f"Кількість обчислень функції для A1: {result_A1['func_calls']}")


A0 = np.array([-1.5, -1.5])
A1 = np.array([3.0, 1.5])

coord_descent = CoordinateDescent()
result_A0_coord = coord_descent.optimize(A0)
coord_descent = CoordinateDescent()
result_A1_coord = coord_descent.optimize(A1)
plot_optimization('координатного спуску', A0, A1, result_A0_coord, result_A1_coord)

# Найшвидший спуск
grad_descent = GradientDescent()
result_A0_grad = grad_descent.optimize(A0)
grad_descent = GradientDescent()
result_A1_grad = grad_descent.optimize(A1)
plot_optimization('найшвидшого спуску', A0, A1, result_A0_grad, result_A1_grad)