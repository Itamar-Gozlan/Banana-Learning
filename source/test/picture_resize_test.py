import os
from shutil import rmtree

from utils import *
from picture_manipulation import *

normalPic = 'normal_test.jpg'
thermalPic = 'thermal_test.JPG'

'''tests start '''
def image_load_test():
    try:
        image_load(normalPic)
        image_load(thermalPic)
        return True
    except:
        return False


def image_save_test():
    os.mkdir("./save_res")
    try:
        im = image_load(normalPic)
        image_save(im, "./save_res", name="new_normal.jpg")
        image_load("./save_res/new_normal.jpg")
        return True
    except:
        return False
    finally:
        rmtree("./save_res")


def image_make_square_test():
    try:
        image = image_load(normalPic)
        x,y = image.size
        square_size = min(x,y)
        new_image = image_make_square(image)
        if (square_size,square_size) == new_image.size:
            return True
        else:
            return False
    except:
        return False


def image_change_size_test():
    image = image_load(normalPic)
    new_image = image_change_size(image, (512, 512))


'''tests end '''


def res_string(res):
    c = Colors()
    fail = [CT.bold, CT.red]
    success = [CT.green]
    if res is True:
        return c.cs(success, "Success")
    return c.cs(fail, "Fail")


def test_runner():
    print("load test {}".format(res_string(image_load_test())))
    print("image_save test {}".format(res_string(image_save_test())))
    print("image_make_square test {}".format(res_string(image_make_square_test())))
    print("image_make_square test {}".format(res_string(image_change_size_test())))


def main():
    test_runner()


if __name__ == '__main__':
    main()