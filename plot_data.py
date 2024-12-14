import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

data1 = pd.read_csv("Data1.csv")

period1_log = np.log10(data1.pl_orbper)
mass1_log = np.log10(data1.pl_bmassj)
xlim = [-0.5,4]
ylim = [-3,1.5]
xlabel = "$log_{10}(P_{orb}/day)$"
ylabel = "$log_{10}(M_p/M_{Jup})$"

output_dir = "images"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    
def common_function(
    title,
    img_title,
    n,
    color="black",
    x=[],
    y=[],
    k=[],
    mid_y=[],
    i=-10,
    x1=[],
    y1=[],
    y2=[],
):
    if(len(x) > 0 and len(y) > 0):
        # print(x,y)
        plt.plot(x, y, color)
        
    if(len(k) > 0 and len(mid_y) > 0):
        for i in range(len(mid_y)+1):
            x1 = np.linspace(-0.5,4,500)
            y1 = np.repeat(k[i], 500)
            plt.plot(x1,y1,"-")
            
    if len(x1) > 0 and len(y1) > 0 and len(y2) > 0:
        plt.plot(x1, y1)
        plt.plot(x1, y2)
    
    plt.scatter(period1_log, mass1_log, s=1, c='#26495c', marker='o')
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    if n == -1:
        fig = plt.savefig(os.path.join(output_dir, img_title + ".png"), format="png", bbox_inches="tight")
    elif n == "mean":
        fig = plt.savefig(os.path.join(output_dir, img_title + "_" + str(i+1) + ".png"), format="png", bbox_inches="tight") 
        plt.show()
    elif n == "final":
        fig = plt.savefig(os.path.join(output_dir, img_title + ".png"), format="png", bbox_inches="tight") 
        plt.show()
    else:
        if i == -10:
            fig = plt.savefig(os.path.join(output_dir, "Obs_" + str(n+1) + "_" + img_title + ".png"), format="png", bbox_inches="tight") 
        else:
            fig = plt.savefig(os.path.join(output_dir, "Obs_" + str(n+1) + "_" + img_title + "_" + str(i+1) + ".png"), format="png", bbox_inches="tight")  
    
    # plt.show()
    plt.close(fig)
    
    
#  Plotting Boundary Bin
def plot_boundary_bin(
    points_x,
    points_y,
    mid_y,
    title,
    k,
    n,
    sd_x=[],
    sd_y=[],
):
    plt.scatter(period1_log, mass1_log, s=1, c='#26495c', marker='o')
    plt.scatter(points_x, points_y, s=1, c='black', marker='o')
    for i in range(len(mid_y)+1):
        x1 = np.linspace(-0.5,4,500)
        y1 = np.repeat(k[i], 500)
        plt.plot(x1,y1,"-")
    # if(len(sd_x) > 0 and len(sd_y) > 0):
    #     plt.fill_between(np.array(sd_y).flatten(), np.array(points_x - sd_x).flatten(), np.array(points_x + sd_x).flatten(),  where=None, interpolate=False, alpha=0.3, label="Std Deviation", color="red" )

    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    if n == "final":
        fig = plt.savefig(os.path.join(output_dir, "mean_final_boundary.png"), format="png", bbox_inches="tight")
        plt.show()
    else:
        fig = plt.savefig(os.path.join(output_dir, "Obs_" + str(n+1) + "_boundary.png"), format="png", bbox_inches="tight")
    # plt.show()
    plt.close(fig)
    
 