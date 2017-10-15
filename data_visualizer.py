import matplotlib.pyplot as plt
import os
import numpy as np

def createNewFolder(folder_path):
    #Create folder for saving figure
    if not os.path.isdir(folder_path):
        os.mkdir(folder_path)

def draw_plot(listX, listY, labelX, labelY, title, save_folder_path, file_name, dpi = 150):
    createNewFolder(save_folder_path)

    plt.figure(dpi=dpi)
    plt.plot(listX, listY)
    plt.xlabel(labelX)
    plt.ylabel(labelY)
    plt.title(title)
    plt.savefig(os.path.join(save_folder_path, file_name))
    plt.show()

def draw_multi_plot(listX, dict_listY, labelX, labelY, list_plot_label, prefix_plot_label, title, save_folder_path, file_name, dpi = 150):
    createNewFolder(save_folder_path)

    plt.figure(dpi=dpi)

    for plot_label in list_plot_label:
        plt.plot(listX, dict_listY[plot_label], label=str(prefix_plot_label)+str(plot_label))
    
    plt.legend()
    plt.xlabel(labelX)
    plt.ylabel(labelY)
    plt.title(title)
    plt.savefig(os.path.join(save_folder_path, file_name))
    plt.show()
