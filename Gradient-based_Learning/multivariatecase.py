

def f(x1,x2) :
    return x1 ** 2 + x2 ** 2

def df_dx1(x1,x2) :
    return 2 * x1

def df_dx2(x1,x2) :
    return 2 * x2

x1 = 3
x2 = 3
ITERATIONS = 10


for iter in range(ITERATIONS):
    dy_dx1 = df_dx1(x1,x2)
    dy_dx2 = df_dx2(x1,x2)
    x1 = x1 - dy_dx1*0.3
    x2 = x2 - dy_dx2*0.3
    print(x1,x2)