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

 # Ploting data1
common_function(
    'Single host Star',
    "data1",
    -1
)


results = []


# range(len(c_l))
for n in range(len(c_l)):
    # c_l[0] 1st line among the 100
    # c_l[n][0] = x coordinate of the curve
    # c_l[n][1] = y coordinate of the curve
    # c_l[n][2] = the 8 segmenst's y coordinate
    # c_l[n][3] = 7 mid point's mid_x and mid_y coordinates of the 8 segments
    # c_l[n][4] = 7 tangents at the 7 mid points
    
    x = c_l[n][0]
    y = c_l[n][1]
    k = c_l[n][2]
    mid_x = c_l[n][3][0]
    mid_y = c_l[n][3][1]
    dydx = c_l[n][4]

    # Ploting initial guess 
    common_function(
        'Single host Star',
        "data1_guess",
        n,
        "gray",
        x,
        y
    )


    #Ploting Segments
    common_function(
        'Single host Star(Segments)',
        "data1_segments",
        n,
        "gray",
        x,
        y,
        k,
        mid_y
    )

    #boundary index and boundary points
    boundary_idx = np.array([])
    # points_x = []
    # points_y = []
    
    result = []

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
            condition_2 = period1_log[j] <= mid_x[i] + 0.5 and period1_log[j] > mid_x[i] - 0.5
            if condition_1 and condition_2:
                idx.append(j)
                
        # print(len(idx))

        # data points within the segments
        P_temp = period1_log[idx].tolist()
        m_temp = mass1_log[idx].tolist()
        plt.scatter(P_temp, m_temp, s=15, c='black', marker='o')

        # perpendicular line to the curve
        l_x = np.linspace(mid_x[i] - 0.5, mid_x[i] + 0.5,100)  # line with a slope of the curve at the center of the segment
        l_y = ((1 / dydx[i]) * (l_x - mid_x[i])) + mid_y[i]  # line with a slope of the curve at the center of the segment
        l_y_per = (-dydx[i] * (l_x - mid_x[i])) + mid_y[i]  # perpendicular line to the straight line at the center of the segment

        def X(x0, d, m):
            x1 = x0 + (d / np.sqrt(1 + m ** 2))
            x2 = x0 - (d / np.sqrt(1 + m ** 2))
            return np.append(x1, x2)


        m = -dydx[i]
        x0 = mid_x[i]
        d = np.array([0, 0.1, 0.2, 0.3, 0.4, 0.5])
        p_x = np.sort(X(x0, d, m))[2:]
        p_x = np.delete(p_x, 3)  # index 3 mid_xas repeated, that's why
        p_y = (-dydx[i] * (p_x - mid_x[i])) + mid_y[i]

        
        # bins lines
        for j in range(8):

            l_y_pera1 = ((1 / dydx[i]) * (l_x - p_x[j])) + p_y[j]
            l_y_pera2 = ((1 / dydx[i]) * (l_x - p_x[j + 1])) + p_y[j + 1]

            plt.plot(l_x, l_y_pera1, "black")
            plt.plot(l_x, l_y_pera2, "black")

            # is within the two consecutive bin-lines: counts
            c = 0
            for e in range(len(idx)):
                if (1 / dydx[i]) > 0:
                    if (m_temp[e] <= (((1 / dydx[i]) * (P_temp[e] - p_x[j])) + p_y[j])) and (
                            m_temp[e] > (((1 / dydx[i]) * (P_temp[e] - p_x[j + 1])) + p_y[j + 1])):
                        c += 1
                else:
                    if (m_temp[e] >= (((1 / dydx[i]) * (P_temp[e] - p_x[j])) + p_y[j])) and (
                            m_temp[e] < (((1 / dydx[i]) * (P_temp[e] - p_x[j + 1])) + p_y[j + 1])):
                        c += 1
            bin_counts.append(c)
        bin_counts = np.array(bin_counts)
        # print(bin_counts)
        percent_bin_counts = (bin_counts / len(idx)) * 100
        # print(percent_bin_counts)
        avg_percent = np.sum(percent_bin_counts) / len(percent_bin_counts)
        # print(avg_percent)
        bins = np.arange(0, 8)

        # Plotting aligned bins
        common_function(
            'Single host Star',
            '1_aligned_bin',
            n,
            "gray",
            x,
            y,
            [],
            [],
            i
        )
        

        # Boundary decision and taking the boundary points
        decision = np.argwhere(np.array(percent_bin_counts) > avg_percent)[0]
        boundary_idx = np.append(boundary_idx, decision)
        b_y = np.linspace(y1[1], y2[1], 50)
        b_x = dydx[i] * (b_y - p_y[decision]) + p_x[decision]
        # points_x.append(b_x)
        # points_y.append(b_y)
        result.append([b_x,b_y])

        # hist ploting and boundary visualisation
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
        fig2 = plt.savefig(os.path.join(output_dir,"Obs_" + str(n+1) + "_1_hist_{numb}.png".format(numb=i+1)), format="png", bbox_inches="tight")   
        # plt.show()
        plt.close(fig2)

        # Plotting Boundary
        common_function(
            "Single host Star",
            "1_boundary_bin",
            n,
            "black",
            b_x,
            b_y,
            [],
            [],
            i,
            x1,
            y1,
            y2
        )
    
        
        
    results.append(result)
    # points_x = np.array(points_x)
    # points_y = np.array(points_y)
    # print(boundary_idx)

    # # Plotting Boundary Bin
    
    # plot_boundary_bin(
    #     points_x,
    #     points_y,
    #     mid_y,
    #     "Single Host Star(Boundary Bins)",
    #     k,
    #     n
    # )
    
    


    # #curvefitting
    # def func(y,a,b,d):
    #     x = a*(y+b)**2 + d
    #     return x

    # A = [[np.sum(np.square(points_y)), np.sum(points_y), len(points_y)],[np.sum(np.power(points_y,3)), np.sum(np.square(points_y)), np.sum(points_y)], [np.sum(np.power(points_y,4)), np.sum(np.power(points_y,3)), np.sum(np.square(points_y))]]
    # B = [np.sum(points_x), np.sum(points_x*points_y), np.sum(points_x*np.square(points_y))]

    # C = np.matmul(np.linalg.inv(A), B)
    # x = C[0]*y**2 + C[1]*y + C[2]
    # # print(C)
    
  

    # #Plotting Result
    # common_function(
    #     "Single Host Star(Boundary)",
    #     "1_result",
    #     n,
    #     "black",
    #     x,
    #     y
    # )
    
    
print("------------------------------------------------------------------------------------------------------------")

r = [[],[],[],[],[],[],[]]
for i in range(len(results)):
    for j in range(7):
        if len(r[j]) == 0:
            r[j].append(results[i][j][0])
            r[j].append(results[i][j][1])
            print(r)
        else:
            r[j][0] = np.array(r[j][0]) + np.array(results[i][j][0])
            r[j][1] = np.array(r[j][1]) + np.array(results[i][j][1])


r = np.array(r)

points_x = []
points_y = []

for i in range(len(r)):
    r[i][0] = r[i][0]/len(results)
    points_x.append(r[i][0])
    points_y.append(r[i][1])
    
    x1 = np.linspace(-0.5, 4, 500)
    y1 = np.repeat(k[i], 500)
    y2 = np.repeat(k[i + 1], 500)
    
    print(r[0][1])

    # Plotting Boundary
    common_function(
        "Single host Star",
        "1_mean_boundary_bin",
        "mean",
        "black",
        r[i][0],
        r[i][1],
        [],
        [],
        i,
        x1,
        y1,
        y2
    )
    
points_x = np.array(points_x)
points_y = np.array(points_y)
    
# Plotting Boundary Bin
   
plot_boundary_bin(
    points_x,
    points_y,
    c_l[0][3][1],
    "Single Host Star(Boundary Bins)",
    c_l[0][2],
    "final"
)


#curvefitting
y = c_l[0][1]
A = [[np.sum(np.square(points_y)), np.sum(points_y), len(points_y)],[np.sum(np.power(points_y,3)), np.sum(np.square(points_y)), np.sum(points_y)], [np.sum(np.power(points_y,4)), np.sum(np.power(points_y,3)), np.sum(np.square(points_y))]]
B = [np.sum(points_x), np.sum(points_x*points_y), np.sum(points_x*np.square(points_y))]
C = np.matmul(np.linalg.inv(A), B)
x = C[0]*y**2 + C[1]*y + C[2]
print(C)


#Plotting Result
common_function(
    "Single Host Star(Boundary)",
    "1_final_result",
    "final",
    "black",
    x,
    y
)


# print(r)


# r = []
# for i in range(len(results)):
#     x = results[i][0]
#     for j in range(len(results[i][0])):
#         if len(r) < j+1:
#             r.append(results[i][0][j])
#         else:
#             r[j] = r[j] + results[i][0][j]
        
# x = np.array(r)/1251
# plt.plot(x, y, "black")
# plt.scatter(period1_log, mass1_log, s=1, c='#26495c', marker='o')
# plt.xlim([-0.5,4])
# plt.ylim([-3,1.5])
# plt.xlabel("xlabel")
# plt.ylabel("ylabel")
# plt.show()
    
    




