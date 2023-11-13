import numpy as np

class BCE :
    def __init__(self,y,pred_y):
        self.y = y
        self.pred_y = pred_y

    def foward(self):
        a = -((self.y*np.log10(self.pred_y))+(1-self.y)*np.log10(1-self.pred_y))
        return a

los = BCE(1,np.arange(start=0.1,stop=1,step=0.1))

print(los.foward())