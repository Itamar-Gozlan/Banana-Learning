###############################################################################
# triplets_gen.py                                                             #
# Technion GIP Final Project                                                  #
# implementation:  Itamar Gozlan                                              #
# This file generates a modified dataset of triplets                          #
# from segmented Bana plants (but can be applied for every dataset)           #
# To use: change src/dst folders and run                                      #
###############################################################################
import cv2
import os
import numpy as np


src_path_A = '/home/itamargoz/data/segmented/SEG_RGB_A'
src_path_B = '/home/itamargoz/data/segmented/SEG_RGB_B'
src_path_C = '/home/itamargoz/data/segmented/SEG_RGB_C'
src_path_D = '/home/itamargoz/data/segmented/SEG_RGB_D'


dst_path_A = '/home/itamargoz/data/segmented/triplets/RGB_A'
dst_path_B = '/home/itamargoz/data/segmented/triplets/RGB_B'
dst_path_C = '/home/itamargoz/data/segmented/triplets/RGB_C'
dst_path_D = '/home/itamargoz/data/segmented/triplets/RGB_D'

src_paths = [src_path_A] #, src_path_B, src_path_C, src_path_D]
dst_paths = [dst_path_A] #, dst_path_B, dst_path_C, dst_path_D]

# src_paths = [src_path_B, src_path_C, src_path_D]
# dst_paths = [dst_path_B, dst_path_C, dst_path_D]


def crate_target_name(src_a, src_b, src_c):
    a = src_a.split("_")
    b = src_b.split("_")
    c = src_c.split("_")
    name = ""
    for it in zip(a, b, c):
        name_set = np.unique([x for x in it])
        curr_substr = ''.join(e + "_" for e in name_set)
        name = name + curr_substr
    name = name[:-1]
    return name


def join_pictures(src_path, dst_path, images):
    print(("========== Join picutres =========="))
    dim = (336, 252)
    for i in range(2, len(images)):
        new_name = crate_target_name(images[i - 2], images[i - 1], images[i])
        if os.path.exists(dst_path + "/" + new_name):
            print(dst_path + "/" + new_name," exist! skipping")
            next
        try:
            image1 = cv2.imread(src_path + "/" + images[i - 2])
            image2 = cv2.imread(src_path + "/" + images[i - 1])
            image3 = cv2.imread(src_path + "/" + images[i])
            image1 = cv2.resize(image1, dim)
            image2 = cv2.resize(image2, dim)
            image3 = cv2.resize(image3, dim)

            target_img = cv2.hconcat([image1, image2, image3])
            write_status = cv2.imwrite(dst_path + "/" + new_name, target_img)
            print("saving ", dst_path + "/" + new_name, " status = ", str(write_status))
        except:
            print(dst_path + "/" + new_name," ERROR! skipping")


def organize_and_execute_join(src_path, dst_path):
    images = os.listdir(src_path)
    aux_ids = [x.split("_")[6] for x in images]
    ids = np.unique(aux_ids)
    aux_arr = [''.join(reversed(x)) for x in images]
    aux_arr.sort()
    images = [''.join(reversed(x)) for x in aux_arr]
    for id in ids:
        print("===== Now processing ID: ", id, " =====")
        result = list(filter(lambda x: (x.split("_")[6] == id), images))
        result.sort(key=lambda x: int(x.split("_")[1]))
        join_pictures(src_path, dst_path, result)


for src_path, dst_path in zip(src_paths, dst_paths):
    if not os.path.exists(dst_path):
        os.mkdir(dst_path)
    print("=== BEGIN ===")
    organize_and_execute_join(src_path, dst_path)
    print("=== END ===")
