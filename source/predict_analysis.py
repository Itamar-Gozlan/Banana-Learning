from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
import numpy as np


CIFAR10_PATH = "C:/Users/ItamarGIP/PycharmProjects/Banana-Learning/saved_models/CIFAR-10.h5"
model = load_model(CIFAR10_PATH)


def get_data_gen(use_augmentation):
    if use_augmentation:
        return ImageDataGenerator(rotation_range=70,  # randomly rotate images in the range (degrees, 0 to 180)
                                      horizontal_flip=True,
                                      vertical_flip=True,
                                      # randomly shift images horizontally (fraction of total width)
                                      width_shift_range=0.2,
                                      # randomly shift images vertically (fraction of total height)
                                      height_shift_range=0.2,
                                      shear_range=0.2) # set range for random shear)
    else:
        return ImageDataGenerator()

def gen_validation_iter(shape: tuple, use_augmentation):
    data_gen = get_data_gen(use_augmentation)