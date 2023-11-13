import numpy as np
import matplotlib.pyplot as plt

def f(x1,x2) : return x1**2 + x2**2


def function_x1(x1,x2) :
    return 2 * x1

def function_x2(x1,x2) :
    return  x2

x1,x2 = 3,3
ITERATIONS = 20

x1_arr, x2_arr = [],[]

for _ in range(ITERATIONS):
    dy_dx1 = function_x1(x1,x2)
    dy_dx2 = function_x2(x1,x2)
    x1 = x1 - dy_dx1*0.978
    x1_arr.append(x1)
    x2 = x2 - dy_dx2*0.1
    x2_arr.append(x2)
    print(x1,x2)

function_x1 = np.linspace(-5,5,100)
function_x2 = np.linspace(-5,5,100)

function_X1,function_X2 = np.meshgrid(function_x1,function_x2)
function_Y = np.log(f(function_X1,function_X2))

fig, ax = plt.subplots(figsize=(10,10))
ax.contour(function_X1,function_X2,function_Y,
           levels=100, cmap ='Reds_r')
ax.scatter(x1_arr,x2_arr,color='blue')

ax.set_xlabel("x1",fontsize=15)
ax.set_ylabel("x2",fontsize=15)
ax.tick_params(labelsize=15)
fig.tight_layout()
plt.show()