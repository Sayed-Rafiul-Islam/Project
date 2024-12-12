# -*- coding: utf-8 -*-

# global imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import close
import os


# local imports 
from initial_curve_line import *
from plot_data import *
data1 = pd.read_csv("Data1.csv")


# -----------------------------------------------------------------------------------------

period1_log = np.log10(data1.pl_orbper)
mass1_log = np.log10(data1.pl_bmassj)


#segment deviding
k = [-2.2, -1.9, -1.6, -1.35, -1.15, -0.6, -0.2, 0.2] 
# midpoint (y) for tangent line
mid_y = np.array([(a + b)/2 for a, b in zip(k[0:-1:], k[1:])])

# midpoint (x) for tangent line
def mid_x(mid_y):
    mid_x = -0.6*(mid_y + 0.8)**2 + 0.55 
    return mid_x

#first derivative at the point for tangent line
def dydx(mid_y):
    dydx = -(6/5)*(mid_y + (4/5)) 
    return dydx



# Ploting data1
plot_data1(
    period1_log,
    mass1_log,
    [-0.5,4],
    [-3,1.5],
    '$log_{10}(P_{orb}/day)$',
    '$log_{10}(M_p/M_{Jup})$',
    'Single host Star'
)

# Ploting initial guess 
plot_initial_guess(
    period1_log,
    mass1_log,
    x,
    y,
    [-0.5,4],
    [-3,1.5],
    '$log_{10}(P_{orb}/day)$',
    '$log_{10}(M_p/M_{Jup})$',
    'Single host Star'
)


#Ploting Segments
plot_segments(
    period1_log,
    mass1_log,
    x,
    y,
    [-0.5,4],
    [-3,1.5],
    '$log_{10}(P_{orb}/day)$',
    '$log_{10}(M_p/M_{Jup})$',
    'Single host Star(Segments)',
    k,
    mid_y
)

#boundary index and boundary points
boundary_idx = np.array([])
points_x = []
points_y = []

# parallel segments
for i in range(len(mid_y)):
    # segments ploting
    x1 = np.linspace(-0.5, 4, 500)
    y1 = np.repeat(k[i], 500)
    y2 = np.repeat(k[i + 1], 500)
    plt.plot(x1, y1)
    plt.plot(x1, y2)

    # index inside the segments
    idx = []
    bin_counts = []

    for j in range(len(mass1_log)):
        condition_1 = (mass1_log[j] >= y1[1]) and (mass1_log[j] < y2[1])
        condition_2 = period1_log[j] <= mid_x(mid_y)[i] + 0.5 and period1_log[j] > mid_x(mid_y)[i] - 0.5
        if condition_1 and condition_2:
            idx.append(j)
            
    print(len(idx))

    # data points within the segments
    P_temp = period1_log[idx].tolist()
    m_temp = mass1_log[idx].tolist()
    plt.scatter(P_temp, m_temp, s=15, c='black', marker='o')

    # perpendicular line to the curve
    l_x = np.linspace(h(mid_y)[i] - 0.5, mid_x(mid_y)[i] + 0.5,100)  # line with a slope of the curve at the center of the segment
    l_y = ((1 / dydx(mid_y)[i]) * (l_x - mid_x(mid_y)[i])) + mid_y[i]  # line with a slope of the curve at the center of the segment
    l_y_per = (-dydx(mid_y)[i] * (l_x - mid_x(mid_y)[i])) + mid_y[i]  # perpendicular line to the straight line at the center of the segment

    def X(x0, d, m):
        x1 = x0 + (d / np.sqrt(1 + m ** 2))
        x2 = x0 - (d / np.sqrt(1 + m ** 2))
        return np.append(x1, x2)


    m = -dydx(mid_y)[i]
    x0 = mid_x(mid_y)[i]
    d = np.array([0, 0.1, 0.2, 0.3, 0.4, 0.5])
    p_x = np.sort(X(x0, d, m))[2:]
    p_x = np.delete(p_x, 3)  # index 3 mid_xas repeated, that's why
    p_y = (-dydx(mid_y)[i] * (p_x - mid_x(mid_y)[i])) + mid_y[i]


    # bins lines
    for j in range(8):

        l_y_pera1 = ((1 / dydx(mid_y)[i]) * (l_x - p_x[j])) + p_y[j]
        l_y_pera2 = ((1 / dydx(mid_y)[i]) * (l_x - p_x[j + 1])) + p_y[j + 1]

        plt.plot(l_x, l_y_pera1, "black")
        plt.plot(l_x, l_y_pera2, "black")

        # is within the two consecutive bin-lines: counts
        c = 0
        for e in range(len(idx)):
            if (1 / dydx(mid_y)[i]) > 0:
                if (m_temp[e] <= (((1 / dydx(mid_y)[i]) * (P_temp[e] - p_x[j])) + p_y[j])) and (
                        m_temp[e] > (((1 / dydx(mid_y)[i]) * (P_temp[e] - p_x[j + 1])) + p_y[j + 1])):
                    c += 1
            else:
                if (m_temp[e] >= (((1 / dydx(mid_y)[i]) * (P_temp[e] - p_x[j])) + p_y[j])) and (
                        m_temp[e] < (((1 / dydx(mid_y)[i]) * (P_temp[e] - p_x[j + 1])) + p_y[j + 1])):
                    c += 1
        bin_counts.append(c)
    bin_counts = np.array(bin_counts)
    print(bin_counts)
    percent_bin_counts = (bin_counts / len(idx)) * 100
    print(percent_bin_counts)
    avg_percent = np.sum(percent_bin_counts) / len(percent_bin_counts)
    print(avg_percent)
    bins = np.arange(0, 8)

    # Plotting aligned bins
    plot_aligned_segments(
        period1_log,
        mass1_log,
        x,
        y,
        [-0.5,4],
        [-3,1.5],
        '$log_{10}(P_{orb}/day)$',
        '$log_{10}(M_p/M_{Jup})$',
        'Single host Star',
        i
    )
    

    # Boundary decision and taking the boundary points
    decision = np.argwhere(np.array(percent_bin_counts) > avg_percent)[0]
    boundary_idx = np.append(boundary_idx, decision)
    b_y = np.linspace(y1[1], y2[1], 50)
    b_x = dydx(mid_y)[i] * (b_y - p_y[decision]) + p_x[decision]
    points_x.append(b_x)
    points_y.append(b_y)

    # mid_xist ploting and boundary visualisation
    fig, ax = plt.subplots()
    width = 1  # the width of the bars
    rects1 = ax.bar(bins - width / 2, bin_counts, width, align='edge', fc='skyblue', ec='black')
    plt.text(.01, .99, "Threshold: {thres}%".format(thres=np.round(avg_percent, 2)), ha='left', va='top',
             transform=ax.transAxes)
    plt.text(.01, .94, "Boundary bin's index: {num}".format(num=int(decision)), ha='left', va='top',
             transform=ax.transAxes)
    y_l = np.linspace(0, bin_counts[decision], 5)
    x_l = np.repeat(decision, len(y_l)) - width / 2
    ax.plot(x_l, y_l, "-o", color="#27445C")
    ax.set_xticks(bins)
    ax.set_ylabel('Number of Planets')
    ax.set_xlabel('Bin Index')
    ax.set_title('Deciding boundary bin')
    ax.bar_label(rects1, labels=["{perc}%".format(perc=letter) for letter in np.round(percent_bin_counts, 2)],
                 padding=0, color='#000080', fontsize=8)
    fig2 = plt.savefig(os.path.join(output_dir, "1_hist_{numb}.png".format(numb=i)), format="png", bbox_inches="tight")   
    plt.show()
    plt.close(fig2)

    # Plotting Boundary
    plot_boundary(
        period1_log,
        mass1_log,
        b_x,
        b_y,
        x1,
        y1,
        y2,
        [-0.5,4],
        [-3,1.5],
        "$log_{10}(P_{orb}/day)$",
        "$log_{10}(M_p/M_{Jup})$",
        "Single host Star",
        i
    )
    
points_x = np.array(points_x)
points_y = np.array(points_y)
print(boundary_idx)

# Plotting Boundary Bin

print(k)

plot_boundary_bin(
       period1_log,
       mass1_log,
       points_x,
       points_y,
       mid_y,
       x1,
       y1,
       [-0.5,4],
       [-3,1.5],
       "$log_{10}(P_{orb}/day)$",
       "$log_{10}(M_p/M_{Jup})$",
       "Single Host Star(Boundary Bins)",
       k
   )


#curvefitting
def func(y,a,b,d):
    x = a*(y+b)**2 + d
    return x

A = [[np.sum(np.square(points_y)), np.sum(points_y), len(points_y)],[np.sum(np.power(points_y,3)), np.sum(np.square(points_y)), np.sum(points_y)], [np.sum(np.power(points_y,4)), np.sum(np.power(points_y,3)), np.sum(np.square(points_y))]]
B = [np.sum(points_x), np.sum(points_x*points_y), np.sum(points_x*np.square(points_y))]

C = np.matmul(np.linalg.inv(A), B)
x = C[0]*y**2 + C[1]*y + C[2]
print(C)

#Plotting Result
plot_result(
       period1_log,
       mass1_log,
       x,
       y,
       [-0.5,4],
       [-3,1.5],
       "$log_{10}(P_{orb}/day)$",
       "$log_{10}(M_p/M_{Jup})$",
       "Single Host Star(Boundary)",
   )


# ------------------------------------------------------------- Questions



