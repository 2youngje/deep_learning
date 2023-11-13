import numpy as np
class AffineFuntion:
    def __init__(self,w,b):
        self.w = w
        self.b = b

    def __call__(self,x) :
        z = np.dot(self.w,x)+self.b
        return z

affine1 = AffineFuntion(w=[1,-1],b=-1.5)
affine2 = AffineFuntion(w=[-1,-1],b=0.5)
