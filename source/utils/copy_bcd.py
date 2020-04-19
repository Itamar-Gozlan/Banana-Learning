# get all validation
# get all test
# get all train
# take shuffle from each list
# combine lists
# move them to bcd_tratment

import os
from shutil import copyfile
import random
import math

global_src = "/home/itamargoz/data/sorted/seg"
global_dst = "/home/itamargoz/data/sorted/A_VS_ALL"


def copy_with_threshold(src, dst, th):
    arr = [os.path.join(src, filename) for filename in os.listdir(src)]
    random.shuffle(arr)
    for i in range(0, th):
        print(arr[i] + " => " + dst + "/" + arr[i].split("/")[-1])
        copyfile(arr[i], dst + "/" + arr[i].split("/")[-1])


def make_full_path(path):
    sub_path = path.split("/")
    curr_path = ""
    for partial_path in sub_path[1:]:
        curr_path = curr_path + "/" + partial_path
        if not os.path.exists(curr_path):
            print("creating: " + curr_path)
            os.mkdir(curr_path)


category = ["train", "test", "validation"]
category_size = [122, 30, 17]  # 366/3, 91/3, 51/3
classes = ["treat_b", "treat_c", "treat_d"]
for ct, sz in zip(category, category_size):
    for cl in classes:
        curr_dst = global_dst + "/" + ct + "/" + "treat_bcd"
        curr_src = global_src + "/" + ct + "/" + cl
        make_full_path(curr_dst)
        if cl == "treat_d" and ct == "test":
            copy_with_threshold(curr_src, curr_dst, sz + 1)
        else:
            copy_with_threshold(curr_src, curr_dst, sz)  

