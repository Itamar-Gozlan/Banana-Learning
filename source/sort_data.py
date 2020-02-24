#!/usr/bin/python
"""
This part target is to sort the segmented plants into train/validate/test
The validation susbset is by choice only plants with 05,15,25 in their ID from each tratment (category)
The test\train is divided with 0.2/0.8 respectively and randomly
test/train/validation
treat_a/  treat_b/  treat_c/  treat_d/

"""

import os
from shutil import copyfile
import random


dst = "D:/Users Data/ItamarGIP/Desktop/Itamar/data/seg_data/sorted_whole"
path_A = "D:/Users Data/ItamarGIP/Desktop/Itamar/data/seg_data/sorted_whole/plants_RGB_A"
path_B = "D:/Users Data/ItamarGIP/Desktop/Itamar/data/seg_data/sorted_whole/plants_RGB_B"
path_C = "D:/Users Data/ItamarGIP/Desktop/Itamar/data/seg_data/sorted_whole/plants_RGB_C"
path_D = "D:/Users Data/ItamarGIP/Desktop/Itamar/data/seg_data/sorted_whole/plants_RGB_D"

def is_validate_id(id):
    if id == "05" or id == "15" or id == "25":
        return True
    return False


def sort_for_category(dst, src, category):
    sub_roots_dir = [dst + "/validation/"+category, dst + "/test/"+category, dst + "/train/"+category]
    for dir in sub_roots_dir:
        if not os.path.exists(dir):
            os.mkdir(dir)
    arr = []
    for filename in os.listdir(src):
        id = filename.split("_")[6]
        if is_validate_id(id):
            curr_dst = dst+"/validation/"+category+"/"+filename
            # print(curr_dst)
            # print(src+"/"+filename)
            # copyfile(src+"/"+filename, curr_dst)
        else:
            print(filename)
            arr += filename # TODO - fix wrong addition to array

    print(arr)
    random.shuffle(arr)
    print(("===================\n\n\n\==============="))
    print(arr)
    # count = 0
    # for filename in arr:
    #     if count < len(arr)*0.8:
    #         curr_dst = dst + "/train/" + category + "/" + filename
    #         print(curr_dst)
    #         print(src + "/" + filename)
    #         # copyfile(src + "/" + filename, curr_dst)
    #     else:
    #         curr_dst = dst + "/test/" + category + "/" + filename
    #         print(curr_dst)
    #         print(src + "/" + filename)
    #         # copyfile(src + "/" + filename, curr_dst)


root_dirs = [dst+"/validation/", dst+"/test/", dst+"/train/"]

for dir in root_dirs:
    if not os.path.exists(dir):
        os.mkdir(dir)

categories = ["treat_a"] #, "treat_b", "treat_c", "treat_d"]
paths = [path_A, path_B, path_C, path_D]

for category, path in zip(categories, paths):
    print(category," : ",path)
    sort_for_category(dst, path, category)
