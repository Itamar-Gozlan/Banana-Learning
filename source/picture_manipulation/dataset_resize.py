import os

from shutil import copyfile

from picture_manipulation import *

from utils import *


def directory_handler(path, dirnames ):
    '''
    :param path: path to create new directories
    :param dirnames: names of directories to create
    :return: none
    '''
    for directory in dirnames:
        os.mkdir(path + '/' + directory)


def picture_handler(src, dest, name, pic_size):
    im = image_load(os.path.join(src, name))
    im = image_change_size(im, pic_size)
    image_save(im, dest, name=name.replace('.jpg', '').replace('.JPG', '') + '_square.jpg')


def file_handler(src, dest, filenames, pic_size):
    for name in filenames:
        if name.find(".jpg") >= 0 or name.find("JPG") >= 0:
            picture_handler(src,dest,name,pic_size)
        else:
            copyfile(os.path.join(src,name), os.path.join(dest,name)) # regular files (JSON,txt) are copied as is


def dataset_resize(load_path,save_path,picture_size):

    os.mkdir(save_path + '/' + 'banana_Root')
    save_path = os.path.join(save_path, 'banana_Root')

    for dirpath, dirnames, filenames in os.walk(load_path):
        relative_path = dirpath.replace(load_path, '')
        print('working on directory: {}'.format(relative_path))
        dest = os.path.join(save_path, relative_path)
        directory_handler(dest, dirnames)
        file_handler(dirpath, dest, filenames, picture_size)


def main():
    '''
    this is an environment variable as we are working on different machines
    see readme for info on how to set it
    '''
    load_path = os.environ['load_picture_root_path']
    save_path = os.environ['save_picture_root_path']

    c = Colors()
    hdr = [CT.header]
    arg = [CT.yellow]
    print(c.cs(hdr, 'Started re-sizing the dataset'))
    print(c.cs([CT.green], "Load path is: ") + c.cs(arg, load_path))
    print(c.cs([CT.green], "Save path is: ") + c.cs(arg, save_path))

    picture_size = (512, 512)

    dataset_resize(load_path, save_path, picture_size)


if __name__ == '__main__':
    main()


# TODO: this is tested only on normal photo hierarchy, as I am not
#       currently using the thermal images
