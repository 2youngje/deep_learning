import numpy as np
import matplotlib.pyplot as plt

def f(x):
    return 2 * x ** 2


def df_dx(x):
    return 4 * x

x = 3
ITERATIONS = 20
x_track, y_track = [x],[f(x)]

print(f"Intial x: {x}")
for iter in range(ITERATIONS):
    dy_dx = df_dx(x)
    x= x -dy_dx * 0.9


    x_track.append(x)
    y_track.append(f(x))
    print(f"{iter + 1}-th x: {x:.4f}")

fig, axes = plt.subplots(2,1,figsize=(10,5))

function_x = np.linspace(-5,5,100)
function_y = f(function_x)

axes[0].plot(function_x,function_y)
axes[0].scatter(x_track,y_track,
                c=range(ITERATIONS+1),cmap='rainbow')
axes[0].set_xlabel("x",fontsize=15)
axes[0].set_ylabel("y",fontsize=15)

axes[1].plot(function_x,function_y)
axes[1].plot(x_track,marker='o')
axes[1].set_xlabel("Iteration", fontsize=15)
axes[1].set_ylabel("y",fontsize=15)

fig.tight_layout()
plt.show()