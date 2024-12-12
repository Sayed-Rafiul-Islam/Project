import os
import matplotlib.pyplot as plt
import numpy as np

output_dir = "images"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Ploting data1
def plot_data1(
    period1_log,
    mass1_log,
    xlim,
    ylim,
    xlabel,
    ylabel,
    title
):
    plt.scatter(period1_log, mass1_log, s=1, c='#26495c', marker='o')
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    fig = plt.savefig(os.path.join(output_dir, "data1.png"), format="png", bbox_inches="tight")   
    plt.show()
    plt.close(fig)
    
    
# Ploting initial guess 
def plot_initial_guess(
    period1_log,
    mass1_log,
    x,
    y,
    xlim,
    ylim,
    xlabel,
    ylabel,
    title
):
    plt.scatter(period1_log, mass1_log, s=1, c='#26495c', marker='o')
    plt.plot(x, y, "black")
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    fig = plt.savefig(os.path.join(output_dir, "initial_guess.png"), format="png", bbox_inches="tight")   
    plt.show()
    plt.close(fig)
    

# Ploting Segments
def plot_segments(
    period1_log,
    mass1_log,
    x,
    y,
    xlim,
    ylim,
    xlabel,
    ylabel,
    title,
    k,
    mid_y
):
    for i in range(len(mid_y)+1):

        x1 = np.linspace(-0.5,4,500)
        y1 = np.repeat(k[i], 500)

        plt.plot(x1,y1,"-")
        
    plt.scatter(period1_log, mass1_log, s=1, c='#26495c', marker='o')
    plt.plot(x, y, "black")
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    fig = plt.savefig(os.path.join(output_dir, "data1_segments.png"), format="png", bbox_inches="tight")   
    plt.show()
    plt.close(fig)