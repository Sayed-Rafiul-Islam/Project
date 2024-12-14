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
    n
):
    plt.scatter(period1_log, mass1_log, s=1, c='#26495c', marker='o')
    plt.scatter(points_x, points_y, s=1, c='black', marker='o')
    for i in range(len(mid_y)+1):
        x1 = np.linspace(-0.5,4,500)
        y1 = np.repeat(k[i], 500)
        plt.plot(x1,y1,"-")

    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    if n == "final":
        fig = plt.savefig(os.path.join(output_dir, "mean_sd_final_boundary.png"), format="png", bbox_inches="tight")
        # plt.show()
    else:
        fig = plt.savefig(os.path.join(output_dir, "Obs_" + str(n+1) + "_boundary.png"), format="png", bbox_inches="tight")
    # plt.show()
    plt.close(fig)
    

def final_curve(
    # curves,
    mean_curve,
    std_x_curve,
    y,
    title,
    img_title
):
    plt.fill_betweenx(y,
                    mean_curve[0]+std_x_curve,
                    mean_curve[0]-std_x_curve,
                    color='red', alpha=0.3,
                    label='Standard Deviation'
                    )
    
    plt.plot(mean_curve[0], mean_curve[1], "black", label='Mean') 
    plt.scatter(period1_log, mass1_log, s=1, c='#26495c', marker='o')
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.title(title)

    fig = plt.savefig(os.path.join(output_dir, img_title + ".png"), format="png", bbox_inches="tight")
    
    plt.show()
    # plt.close(fig)
    
    
 