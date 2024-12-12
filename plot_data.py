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

# Plotting aligned bins
def plot_aligned_segments(
    period1_log,
    mass1_log,
    x,
    y,
    xlim,
    ylim,
    xlabel,
    ylabel,
    title,
    i
):
    plt.scatter(period1_log, mass1_log, s=1, c='#26495c', marker='o')
    plt.plot(x, y, "black")
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    fig = plt.savefig(os.path.join(output_dir, "1_aligned_bin_{numb}.png".format(numb=i)), format="png", bbox_inches="tight")   
    plt.show()
    plt.close(fig)
    
    
# Plotting Boundary
def plot_boundary(
    period1_log,
    mass1_log,
    x,
    y,
    x1,
    y1,
    y2,
    xlim,
    ylim,
    xlabel,
    ylabel,
    title,
    i
):
    plt.scatter(period1_log, mass1_log, s=1, c='#26495c', marker='o')
    plt.plot(x, y, "black")
    plt.plot(x1, y1)
    plt.plot(x1, y2)
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    fig = plt.savefig(os.path.join(output_dir, "1_boundary_bin_{numb}.png".format(numb=i)), format="png", bbox_inches="tight")   
    plt.show()
    plt.close(fig)
    
    
#  Plotting Boundary Bin
def plot_boundary_bin(
    period1_log,
    mass1_log,
    points_x,
    points_y,
    mid_y,
    x1,
    y1,
    xlim,
    ylim,
    xlabel,
    ylabel,
    title,
    k
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
    fig = plt.savefig(os.path.join(output_dir, "boundary.png"), format="png", bbox_inches="tight")
    plt.show()
    plt.close(fig)
    
    
# Plotting Result
def plot_result(
    period1_log,
    mass1_log,
    x,
    y,
    xlim,
    ylim,
    xlabel,
    ylabel,
    title,
):
    plt.scatter(period1_log, mass1_log, s=1, c='#26495c', marker='o')
    plt.plot(x, y, "black")
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    fig = plt.savefig(os.path.join(output_dir, "1_result.png"), format="png", bbox_inches="tight")   
    plt.show()
    plt.close(fig)
 
    
