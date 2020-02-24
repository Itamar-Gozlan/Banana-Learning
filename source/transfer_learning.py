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

num_epochs = 400
save_folder_path = "C:/Users/ItamarGIP/PycharmProjects/Banana-Learning/saved_models"

"""" --- common utils for all networks --- """


def count_files_in_path(path):
    return sum([len(files) for r, d, files in os.walk(path)])


def get_data_gen(preprocess_input, use_augmentation):
    if use_augmentation:
        return ImageDataGenerator(rotation_range=70,  # randomly rotate images in the range (degrees, 0 to 180)
                                  horizontal_flip=True,
                                  vertical_flip=True,
                                  # randomly shift images horizontally (fraction of total width)
                                  width_shift_range=0.2,
                                  # randomly shift images vertically (fraction of total height)
                                  height_shift_range=0.2,
                                  shear_range=0.2,  # set range for random shear
                                  preprocessing_function=preprocess_input)
    else:
        return ImageDataGenerator(preprocessing_function=preprocess_input)


def gen_iterators(shape: tuple, preprocess_input, use_augmentation):
    data_gen = get_data_gen(preprocess_input, use_augmentation)

    train_it = data_gen.flow_from_directory('D:/Users Data/ItamarGIP/Desktop/Itamar/data/sorted/train',
                                            target_size=shape,
                                            color_mode='rgb',
                                            batch_size=32,
                                            class_mode='categorical',
                                            shuffle=True)

    test_it = data_gen.flow_from_directory('D:/Users Data/ItamarGIP/Desktop/Itamar/data/sorted/test',
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
    validation_gt = data_gen.flow_from_directory('D:/Users Data/ItamarGIP/Desktop/Itamar/data/sorted/validation',
                                                 target_size=(336, 252),
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
    train_size = count_files_in_path('D:/Users Data/ItamarGIP/Desktop/Itamar/data/sorted/train')
    test_size = count_files_in_path('D:/Users Data/ItamarGIP/Desktop/Itamar/data/sorted/test')

    # Mobilenet
    print("============== Mobilenet ==============")
    train_gt, test_gt = gen_iterators((336, 252), mobilenet_preprocess_input, use_augmentation)
    mobilenet_model = MobileNet(weights='imagenet',
                                include_top=False,
                                input_shape=(336, 252, 3))

    model = transfer_learning(train_size, test_size, train_gt, test_gt, mobilenet_model)  # build model
    evaluate_model(model, mobilenet_preprocess_input)
    save__model(model, "MobileNet")
    #
    # # VGG
    # print("============== VGG ==============")
    # train_gt, test_gt = gen_iterators((336, 252), vgg16_preprocess_input, use_augmentation)
    # vgg16_model = VGG16(weights='imagenet',
    #                     include_top=False,
    #                     input_shape=(336, 252, 3))
    #
    # model = transfer_learning(train_size, test_size, train_gt, test_gt, vgg16_model)  # build model
    # evaluate_model(model, vgg16_preprocess_input)
    # save__model(model, "VGG16")

    # GoogLeNet (e.g. InceptionV3).
    print("============== GoogLeNet ==============")

    train_gt, test_gt = gen_iterators((336, 252), gn_preprocess_input, use_augmentation)
    googlenet_model = InceptionV3(weights='imagenet',
                                  include_top=False,
                                  input_shape=(336, 252, 3))

    model = transfer_learning(train_size, test_size, train_gt, test_gt, googlenet_model)  # build model
    evaluate_model(model, gn_preprocess_input)
    save__model(model, "GoogleNet")

    # # ResNet50 (e.g. InceptionV3).
    # print("============== ResNet50 ==============")
    # train_gt, test_gt = gen_iterators((336, 252), resnet_preprocess_input,use_augmentation)
    # resnet_model = ResNet50(weights='imagenet',
    #                         include_top=False,
    #                         input_shape=(336, 252, 3))
    #
    # resnet_model = transfer_learning(train_size, test_size, train_gt, test_gt, googlenet_model)  # build model
    # evaluate_model(model, resnet_preprocess_input)
    # save__model(model, "ResNet50")


# sys.stdout = open('logs/transfer_learning.log', "w")
# execute_all_models(True)


