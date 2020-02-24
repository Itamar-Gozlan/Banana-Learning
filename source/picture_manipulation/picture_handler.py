from PIL import Image


def image_load(path):
    '''
    :param path: path to picture. can be relative or absolute
    :return: new image loaded from path
    '''
    image = Image.open(path)
    # image.show()
    return image


def image_save(image, path, name=''):
    '''
    :param image:  type Image of PIL created with load_image
    :param path: path to save picture. can be relative or absolute
    :param name: new name for the image
    :return: none
    '''
    image.save(path + '/' + name, "jpeg")


def image_make_square(image):
    '''
    :param image: type Image of PIL created with load_image
    :return: new image with square proportions
    '''
    fill_color = (0, 0, 0, 0)
    x, y = image.size
    size = min(x, y)  # to lose miminum data
    new_im = Image.new('RGBA', (size, size), fill_color)
    new_im.paste(image, (int((size - x) / 2), int((size - y) / 2)))
    new_im.rotate(180).show()
    # new_im.show()
    return new_im


def image_change_size(image, size):
    '''

    :param image: type Image of PIL created with load_image
    :param size: tuple (vertical_size, horizontal_size) of positive numbers
    :return: image with new size
    '''
    new_image = image.resize(size)
    # new_image.show()
    return new_image


