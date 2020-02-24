import random
import os

from picture_manipulation import picture_handler


def get_single_dir_photo_list(dirpath, filemanes):
    photo_list = []
    for name in filemanes:
        if name.find(".jpg") >= 0 or name.find("JPG") >= 0:
            photo_abs_path = os.path.join(dirpath, name)
            photo_list.append(photo_abs_path)

    return photo_list


def get_all_photo_list(dirpath):
    photo_list = []
    for dirpath, dirnames, filenames in os.walk(dirpath):
        dir_photo_list = get_single_dir_photo_list(dirpath, filemanes)
        photo_list.extend(dir_photo_list)

    return photo_list


def ir_photo_tag(name):
    index = name.find('_IR_')
    tag = name[index+len('_IR_'): len(name)]
    return tag[0:tag.find('_')]


def normal_photo_tag(name):
    index = name.find('_Treat_')
    tag = name[index + len('_Treat_'): len(name)]
    return tag[0:tag.find('_')]


def get_photo_tag(photo_name):
    name = os.path.basename(os.path.normpath(photo_name))
    if name.find('_IR_') >=0:
        return ir_photo_tag(name)
    return normal_photo_tag(name)


def get_tagged_photos(photolist):
    tagged_photos = []
    for photo in photolist:
        tag = get_photo_tag(photo)
        tagged_photos.append([photo, tag])

    return tagged_photos


#  TODO: not sure if its needed here, will come back to it
# def make_random_fold(photolist, fold_num):
#     pass

