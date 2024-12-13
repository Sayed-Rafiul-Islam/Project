import numpy as np

y  = np.linspace(-2.2,0.2,1251) #boundary initial curve

a = np.random.uniform(-0.8, -0.3, 100)
b = np.random.uniform(0.5, 1.0, 100)
c = np.random.uniform(0.3, 0.7, 100)

#segment deviding
k = [-2.2, -1.9, -1.6, -1.35, -1.15, -0.6, -0.2, 0.2] 

# midpoint (y) for tangent line
mid_y = np.array([(a + b)/2 for a, b in zip(k[0:-1:], k[1:])])
print(mid_y)

def h(y):
    curve_lines_data = []
    for ai, bi, ci in zip(a, b, c):
        x = ai*(y+bi)**2 + ci # curve line
        mid_x = ai*(mid_y + bi)**2 + ci 
        dydx = -(6/5)*(mid_y + (4/5)) 
        curve_lines_data.append([x,y,k,(mid_x,mid_y),dydx])
    
    return curve_lines_data
    # x = -a*(y+b)**2 + c #boundary initial curve
    # return x    


c_l = h(y)
