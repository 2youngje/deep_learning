import numpy as np
import matplotlib.pyplot as plt

def f1(x1):
    return (1 / 10) * (np.square(x - 2))


def df1_dx(x1):
    return (1 / 5) * (x - 2)

def f2(x2):
    return (1 / 8) * (np.square(x - 2))

def df2_dx(x2):
    return (1 / 4) * (x - 2)

def f3(x3):
    return (1 / 6) * (np.square(x - 2))

def df2_dx(x3):
    return (1 / 3) * (x - 2)

x1,x2,x3 = 3
ITERATIONS = 20
x_track1, y_track1 = [x1],[f1(x1)]
x_track2, y_track2 = [x2],[f2(x2)]
x_track3, y_track3 = [x3],[f3(x3)]

print(f"Intial x: {x1}")
for iter in range(ITERATIONS):
    dy_dx = df1_dx(x)
    x= x - dy_dx

    x_track1.append(x)
    y_track1.append(f1(x))
    print(f"{iter + 1}-th x: {x:.4f}")

fig, axes = plt.subplots(3,1,figsize=(10,5))

function_x = np.linspace(-5,5,100)
function_y = f1(function_x)

axes[0].plot(function_x,function_y)
axes[0].scatter(x_track1,y_track1,
                c=range(ITERATIONS+1),cmap='rainbow')
axes[0].set_xlabel("x",fontsize=15)
axes[0].set_ylabel("y",fontsize=15)

axes[1].plot(x_track,marker='o')
axes[1].set_xlabel("Iteration", fontsize=15)
axes[1].set_ylabel("y",fontsize=15)

fig.tight_layout()
plt.show()