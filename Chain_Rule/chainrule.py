import numpy as np


class F1:
    def __call__(self, x):
        self.x = x
        z1 = 1 / x
        return z1

    def backward(self, dz2_dz1):
        dz1_dx = 1
        dz2_dx = dz2_dz1 * dz1_dx
        return dz2_dx


class F2:
    def __call__(self, z1):
        z2 = 1 + z1
        return z2

    def backward(self, dz3_dz2):
        dz2_dz1 = 1
        dz3_dz1 = dz3_dz2 * dz2_dz1
        return dz3_dz1


class F3:
    def __call__(self, z2):
        self.z2 = z2
        z3 = np.exp(z2)
        return z3

    def backward(self, da_dz3):
        dz3_dz2 = self.z2 * np.exp(self.z2)
        da_dz2 = da_dz3 * dz3_dz2
        return da_dz2


class F4:
    def __call__(self, z3):
        a = -z3
        return a

    def backward(self):
        da_dz3 = -1
        return da_dz3


class Sigmoid:
    def __init__(self, x):
        self.x = x
        self.fun1 = F1()
        self.fun2 = F2()
        self.fun3 = F3()
        self.fun4 = F4()

    def __call__(self):
        x = self.x
        z4 = self.fun4(x)
        z3 = self.fun3(z4)
        z2 = self.fun2(z3)
        z1 = self.fun1(z2)
        y = z1

        return y

    def backward(self,x):
        z1 = self.fun1.backward(x)
        z2 = self.fun2.backward(z1)
        z3 = self.fun3.backward(z2)
        z4 = self.fun1.backward(z3)
        y = z4
        return y


sigmoid = Sigmoid(1)
result_forward = sigmoid()
result_backward = sigmoid.backward(1)

print(result_forward)
print(result_backward)