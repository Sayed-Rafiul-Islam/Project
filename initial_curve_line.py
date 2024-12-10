import numpy as np

y  = np.linspace(-2.2,0.2,1251) #boundary initial curve

def h(y):
    x = -0.6*(y+0.8)**2 + 0.55 #boundary initial curve
    return x    

x = h(y)