class Function1:
    def foward(self,x):
        z = x - 2
        return z


    def backward(self,dy_dz):
        dz_dx = 1
        dy_dx = dy_dz * dz_dx
        return dy_dx

class Function2:
    def foward(self,z):
        self.z = z
        y = 2*(z**2)
        return y
    def backward(self):
        dy_dz = 4 * self.z
        return dy_dz

class Function:
    def __init__(self, x):
        self.x = x
        self.fun1 = Function1()
        self.fun2 = Function2()
    def foward(self):
        z = self.fun1.foward(self.x)
        y = self.fun2.foward(z)
        a = self.fun2.backward()

        print(f'x = {self.x} 일 때, f1_forward(x) = ',self.fun1.foward(self.x))
        print(f'f2_foward(x)= {self.fun2.foward(z)}')

        print('f1_backword(x) = ', self.fun1.backward(a))

fun1 = Function(5)

fun1.foward()