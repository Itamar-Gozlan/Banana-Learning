import matplotlib.pyplot as plt
import sys

plt.rcdefaults()
import numpy as np
import matplotlib.pyplot as plt

dates = [20180911, 20180912, 20180913, 20180914, 20180915, 20180916, 20180917, 20180918, 20180920, 20180921,
         20180922,
         20180923, 20180924, 20180925, 20180926, 20180927, 20180928]


def fill_and_sort(y_true, y_false):
    for date in dates:
        if not y_false.get(date):
            y_false[date] = 0
        if not y_true.get(date):
            y_true[date] = 0
    y_true_arr = []
    for key in sorted(y_true.keys()):
        y_true_arr.append(float(y_true[key]))
    y_false_arr = []
    for key in sorted(y_false.keys()):
        y_false_arr.append(float(y_false[key]))
    return y_true_arr, y_false_arr


# TODO - Add table with the number at each day
# create plot of true\false predictions bars
def create_bars(y_true_dict, y_false_dict):
    true_pred, false_pred = fill_and_sort(y_true_dict, y_false_dict)
    fig, ax = plt.subplots(figsize=(18, 10))
    index = np.arange(17)  # num dates
    bar_width = 0.35
    opacity = 0.5

    rects1 = plt.bar(index, false_pred, bar_width,
                     alpha=opacity,
                     color='r',
                     label='False Prediction',
                     edgecolor='black')

    rects2 = plt.bar(index + bar_width, true_pred, bar_width,
                     alpha=opacity,
                     color='g',
                     label='True Prediction',
                     edgecolor='black')
    plt.xlabel('Date')
    plt.ylabel('Prediction Count')
    plt.title('Predictions By Date')
    plt.xticks(index + bar_width, dates)
    plt.legend()
    plt.tight_layout()
    return plt


def plot_line_and_approximation(n_epochs, y, accuracy):
    x = np.arange(n_epochs)
    plt.plot(x, y, label='Original Loss', color='b')
    try:
        coefficients = np.polyfit(x, y, 2)
        poly = np.poly1d(coefficients)
        new_x = np.linspace(x[0], x[-1])
        new_y = poly(new_x)
        plt.plot(x, y, new_x, new_y)
        plt.xlim([x[0] - 1, x[-1] + 1])
        plt.plot(new_x, new_y, label='Approximation Loss', color='r')
    except:
        print("polyfit Failed! figure do not include approximation")
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss by Epochs - ' + accuracy)
    plt.legend()
    return plt


def plot_loss(input_file, save_path):
    f = open(input_file, "r")
    name = input_file.split("/")[-1].split(".")[0]
    loss = []
    count = 0
    fig = 0
    for line in f:
        if "val_loss" in line:
            loss.append(float(line.split(" ")[-4]))
            count = count + 1
        if "Test accuracy" in line:
            plt = plot_line_and_approximation(count, loss, line)
            plt.savefig(save_path + "/" + name + str(fig) + '.png')
            plt.clf()
            # plt.show()
            count = 0
            loss = []
            fig = fig + 1


def plot_bars_from_file(input_file, save_path):
    f = open(input_file, "r")
    name = input_file.split("/")[-1].split(".")[0]

    y_true = {}
    y_false = {}
    fig = 0
    true_flag, false_flag = False, False
    for line in f:
        if "========= TOTAL PREDICTION" in line:
            true_flag, false_flag = False, False
            plt = create_bars(y_true, y_false)
            plt.savefig(save_path + "/" + name + str(fig) + '.png')
            fig = fig + 1
            y_true = {}
            y_false = {}
        if "========= TRUE PREDICTION BY DATE =========" in line:
            true_flag = True
            false_flag = False  # should be False
            continue
        if "========= FALSE PREDICTION BY DATE =========" in line:
            true_flag = False
            false_flag = True
            continue
        if true_flag:
            # remove all spaces and split to key,value
            key, value = "".join(line.split()).split(":")
            y_true[int(key)] = value
        if false_flag:
            # remove all spaces and split to key,value
            key, value = "".join(line.split()).split(":")
            y_false[int(key)] = value


save_path = "/home/itamargoz/trunk/Banana-Learning/source/logs/figures"
# python ./plot_from_log.py [bar | loss] [path]
what_to_plot = sys.argv[1]
input_file = sys.argv[2]

plot_bars_from_file(input_file, save_path) if what_to_plot == "bars" else plot_loss(input_file, save_path)
