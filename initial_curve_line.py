import numpy as np

y  = np.linspace(-2.2,0.2,1251) #boundary initial curve

a = np.linspace(-0.55, -0.65, 100)
b = np.linspace(0.75, 0.85, 100)
c = np.linspace(0.5, 0.6, 100)

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
