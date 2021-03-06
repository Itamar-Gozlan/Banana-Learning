###############################################################################
# transfer_learning.py                                                        #
# Technion GIP Final Project                                                  #
# implementation:  Itamar Gozlan                                              #
###############################################################################
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.applications import MobileNet
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet import preprocess_input as mobilenet_preprocess_input
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input as vgg16_preprocess_input
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.inception_v3 import preprocess_input as gn_preprocess_input
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess_input
import os
import sys

"""" --- globals --- """

num_epochs = 300
# dim = (336, 252, 3) # GPU exhausted
# d2_dim = (336, 252)

dim = (151, 504, 3) # GPU exhausted
d2_dim = (151, 504) # GPU exhausted



save_folder_path = "C:/Users/ItamarGIP/PycharmProjects/Banana-Learning/saved_models"
# path_train = 'D:/Users Data/ItamarGIP/Desktop/Itamar/data/seg_data/sorted_whole/train'
# path_test = 'D:/Users Data/ItamarGIP/Desktop/Itamar/data/seg_data/sorted_whole/test'
# path_validation = 'D:/Users Data/ItamarGIP/Desktop/Itamar/data/seg_data/sorted_whole/validation'

# # triplets path
path_train = 'D:/Users Data/ItamarGIP/Desktop/Itamar/data/seg_data/triplets/train'
path_test = 'D:/Users Data/ItamarGIP/Desktop/Itamar/data/seg_data/triplets/test'
path_validation = 'D:/Users Data/ItamarGIP/Desktop/Itamar/data/seg_data/triplets/validation'

"""" --- common utils for all networks --- """

def count_files_in_path(path):
    return sum([len(files) for r, d, files in os.walk(path)])

# use_augmentation False\True preprocess_input from library
def get_data_gen(preprocess_input, use_augmentation):
    if use_augmentation:
        return ImageDataGenerator(rotation_range=70,  # randomly rotate images in the range (degrees, 0 to 180)
                                  horizontal_flip=False, # do not include with triplets
                                  vertical_flip=True,
                                  # randomly shift images horizontally (fraction of total width)
                                  width_shift_range=0.2,
                                  # randomly shift images vertically (fraction of total height)
                                  height_shift_range=0.2,
                                  shear_range=0.2,  # set range for random shear
                                  preprocessing_function=preprocess_input)
    else:
        return ImageDataGenerator(preprocessing_function=preprocess_input)

# Generates iterators from global paths with\without Kears built-in augmentations preprocess_input from library
def gen_iterators(shape: tuple, preprocess_input, use_augmentation):
    data_gen = get_data_gen(preprocess_input, use_augmentation)

    train_it = data_gen.flow_from_directory(path_train,
                                            target_size=shape,
                                            color_mode='rgb',
                                            batch_size=32,
                                            class_mode='categorical',
                                            shuffle=True)

    test_it = data_gen.flow_from_directory(path_test,
                                           target_size=shape,
                                           color_mode='rgb',
                                           batch_size=32,
                                           class_mode='categorical',
                                           shuffle=True)

    return train_it, test_it


def save__model(model, model_name):
    # path_model = save_folder_path
    # save_dir = os.path.join(os.getcwd(), ) + path_model
    save_dir = save_folder_path
    # Save model and weights
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    model_path = os.path.join(save_dir, model_name + ".h5")
    model.save(model_path)
    print('Saved trained model at %s ' % model_path)


def evaluate_model(model, preprocess_input):
    data_gen = ImageDataGenerator(preprocessing_function=preprocess_input)
    validation_gt = data_gen.flow_from_directory(path_validation,
                                                 target_size=d2_dim,
                                                 color_mode='rgb',
                                                 batch_size=32,
                                                 class_mode='categorical',
                                                 shuffle=True)
    # Score trained model.
    scores = model.evaluate(validation_gt, verbose=1)
    print('Test loss:', scores[0])
    print('Test accuracy:', scores[1])


# Pre-Trained Model as Feature Extractor in Model
"""" --- models --- """

# execute a single pre-defined model
def transfer_learning(train_size, test_size, train_gt, test_gt, base_model):
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(
        x)  # we add dense layers so that the model can learn more complex functions and classify for better results.
    x = Dense(1024, activation='relu')(x)  # dense layer 2
    x = Dense(512, activation='relu')(x)  # dense layer 3
    preds = Dense(4, activation='softmax')(x)  # final layer with softmax activation
    model = Model(inputs=base_model.input, outputs=preds)
    for layer in model.layers[:20]:
        layer.trainable = False
    for layer in model.layers[20:]:
        layer.trainable = True
    # Adam optimizer
    # loss function will be categorical cross entropy
    # evaluation metric will be accuracy
    model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])
    batch_size = 32
    steps_per_epoch = train_size / batch_size
    validation_steps = test_size / batch_size

    model.fit_generator(generator=train_gt,
                        steps_per_epoch=steps_per_epoch,
                        epochs=num_epochs,
                        validation_data=test_gt,
                        validation_steps=validation_steps,
                        workers=4)

    return model


"""
VGG (e.g. VGG16 or VGG19).
GoogLeNet (e.g. InceptionV3).
Residual Network (e.g. ResNet50).
"""


def execute_all_models(use_augmentation=False):
    train_size = count_files_in_path(path_train)
    test_size = count_files_in_path(path_test)

    # Mobilenet
    print("============== Mobilenet_"+str(use_augmentation)+" ==============")
    train_gt, test_gt = gen_iterators(d2_dim, mobilenet_preprocess_input, use_augmentation)
    mobilenet_model = MobileNet(weights='imagenet',
                                include_top=False,
                                input_shape=dim)

    model = transfer_learning(train_size, test_size, train_gt, test_gt, mobilenet_model)  # build model
    evaluate_model(model, mobilenet_preprocess_input)
    save__model(model, "Mobilenet_"+str(use_augmentation))
    #
    # # VGG
    # print("============== VGG ==============")
    # train_gt, test_gt = gen_iterators(d2_dim, vgg16_preprocess_input, use_augmentation)
    # vgg16_model = VGG16(weights='imagenet',
    #                     include_top=False,
    #                     input_shape=dim)
    #
    # model = transfer_learning(train_size, test_size, train_gt, test_gt, vgg16_model)  # build model
    # evaluate_model(model, vgg16_preprocess_input)
    # save__model(model, "VGG16")

    # GoogLeNet (e.g. InceptionV3).
    print("============== GoogLeNet"+str(use_augmentation)+" ==============")

    train_gt, test_gt = gen_iterators(d2_dim, gn_preprocess_input, use_augmentation)
    googlenet_model = InceptionV3(weights='imagenet',
                                  include_top=False,
                                  input_shape=dim)

    model = transfer_learning(train_size, test_size, train_gt, test_gt, googlenet_model)  # build model
    evaluate_model(model, gn_preprocess_input)
    save__model(model, "GoogleNet"+str(use_augmentation))

    # # ResNet50 (e.g. InceptionV3).
    # print("============== ResNet50 ==============")
    # train_gt, test_gt = gen_iterators(d2_dim, resnet_preprocess_input,use_augmentation)
    # resnet_model = ResNet50(weights='imagenet',
    #                         include_top=False,
    #                         input_shape=dim)
    #
    # resnet_model = transfer_learning(train_size, test_size, train_gt, test_gt, googlenet_model)  # build model
    # evaluate_model(model, resnet_preprocess_input)
    # save__model(model, "ResNet50")



print("=============== Transfer Leraning - No Augmentation ==============")
execute_all_models(False)


print("=============== Transfer Leraning - No Augmentation ==============")
execute_all_models(True)


