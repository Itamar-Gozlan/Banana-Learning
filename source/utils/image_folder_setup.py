import os
from shutil import copy
import cv2

data_purpose = ["train", "test", "validation"]
categories = ["treat_a", "treat_b", "treat_c", "treat_d"]


def create_category_dirs(dst_dir):
    for dt in data_purpose:
        new_dir = dst_dir + "/" + dt
        if not os.path.exists(new_dir):
            os.mkdir(new_dir)

    for dt in data_purpose:
        for category in categories:
            new_dir = dst_dir + "/" + dt + "/" + category
            if not os.path.exists(new_dir):
                os.mkdir(new_dir)


def populate_path_list(src_dir):
    path_list = []
    for dir_path, dirs, files in os.walk(src_dir, topdown=False):
        for name in files:
            path_list.append(os.path.join(dir_path, name))
    return path_list


def process_picture(src_path, dst_path):
    print(". ", end="")
    file_name = src_path.split('\\')[-1]
    img = cv2.imread(src_path)
    if img.shape[0] > img.shape[1]:
        img = cv2.transpose(img)
    img = cv2.resize(img, (336, 252))
    full_dst_path = dst_path + '/' + file_name
    cv2.imwrite(full_dst_path, img)


def populate_category_dir(dst_root, category, path_list, p_train, p_test):
    p_validate = 1 - (p_train + p_test)
    if p_validate <= 0:
        raise ValueError('p_train + p_test + p_validate >= 1!')
    train_size = int(len(path_list) * p_train)
    test_size = int(len(path_list) * p_test)
    path_train = dst_root + data_purpose[0] + "/" + category
    path_test = dst_root + data_purpose[1] + "/" + category
    path_validation = dst_root + data_purpose[2] + "/" + category
    # copy train
    for i in range(0, train_size):
        process_picture(path_list[i], path_train)
    # copy train
    for i in range(train_size, train_size + test_size):
        process_picture(path_list[i], path_test)
    # copy validation
    for i in range(train_size + test_size, len(path_list)):
        process_picture(path_list[i], path_validation)


def populate_all(categories, category_dir_list, dst_dir, p_train, p_test):
    print("populate may take a few moments...")
    if len(categories) != len(category_dir_list):
        raise ValueError('len(categories) != len(category_dir_list) - have to be the same!')
    for category, category_dir in zip(categories, category_dir_list):
        curr_path_list = populate_path_list(category_dir)
        populate_category_dir(dst_dir, category, curr_path_list, p_train, p_test)

    print("done")


def image_folder_setup(categories, category_dir_list, dst_dir, p_train=0.7, p_test=0.2):
    create_category_dirs(dst_dir)
    populate_all(categories, category_dir_list, dst_dir, p_train, p_test)


# activation code


# data to change when using other machines

src_dir_path = "D:/Users Data/ItamarGIP/Desktop/Itamar/data/banana_water_stress/"
dst_dir_path = "D:/Users Data/ItamarGIP/Desktop/Itamar/data/sorted/"

dir_a = src_dir_path + "task_256"
dir_b = src_dir_path + "task_257"
dir_c = src_dir_path + "task_258"
dir_d = src_dir_path + "task_259"

category_dir_list = [dir_a, dir_b, dir_c, dir_d]
image_folder_setup(categories, category_dir_list, dst_dir_path)


