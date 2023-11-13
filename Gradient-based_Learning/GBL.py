import numpy as np
import matplotlib.pyplot as plt


def f(x):
    return (1 / 10) * (np.square(x - 2))


def df_dx(x):
    return (1 / 5) * (x - 2)


x = 2.5
ITERATIONS = 20
x_values = []
y_values = []

print(f"Initial x: {x}")
for iter in range(ITERATIONS):
    y = f(x)
    dy_dx = df_dx(x)
    x = x - dy_dx

    x_values.append(x)
    y_values.append(y)

    print(f"{iter + 1}-th x: {x:.4f}, f(x): {y:.4f}")

# Plotting
plt.plot(x_values, y_values, '-o')
plt.title('Gradient Descent Optimization')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.show()