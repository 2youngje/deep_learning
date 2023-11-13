import numpy as np

class AffineFuntion:
    def __init__(self,w,b):
        self.w = w
        self.b = b

    def __call__(self,x) :
        z = np.dot(self.w,x)+self.b

        return z

class SigmoidFuntion:
    def __call__(self, z):
        a = 1 / (1+ np.exp(-z))
        return a

class ArticialNeuron :

    def __init__(self,w,b):
        self.affine = AffineFuntion(w=w, b=b)
        self.activation = SigmoidFuntion()

    def __call__(self,x):
        z = self.affine(x)
        a = self.activation(z)
        return a


