import sys
import numpy as np
import matplotlib.pylab as plt
import tensorflow.keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, GlobalAveragePooling2D
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.applications.mobilenet import preprocess_input
from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix
import os
import sys
# ------------- globals -----------------

path_train = 'D:/Users Data/ItamarGIP/Desktop/Itamar/data/seg_data/sorted_whole/train'
path_test = 'D:/Users Data/ItamarGIP/Desktop/Itamar/data/seg_data/sorted_whole/test'
path_validation = 'D:/Users Data/ItamarGIP/Desktop/Itamar/data/seg_data/sorted_whole/validation'



# ------------- utils -----------------
def count_files_in_path(path):
    return sum([len(files) for r, d, files in os.walk(path)])


def define_optimizier(name, lr):
    opt = {
        "RMS": tensorflow.keras.optimizers.RMSprop(lr=lr, decay=1e-6),
        "SGD": tensorflow.keras.optimizers.SGD(lr=lr, decay=1e-6),
        "ADAM": tensorflow.keras.optimizers.SGD(lr=lr, decay=1e-6)
    }
    return opt.get(name, "Invalid name")


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

def gen_iterators(shape: tuple, use_augmentation):
    data_gen = get_data_gen(use_augmentation)

    train_gt = data_gen.flow_from_directory(path_train,
                                            target_size=shape,
                                            color_mode='rgb',
                                            batch_size=32,
                                            class_mode='categorical',
                                            shuffle=True)

    test_gt = data_gen.flow_from_directory(path_test,
                                           target_size=shape,
                                           color_mode='rgb',
                                           batch_size=32,
                                           class_mode='categorical',
                                           shuffle=True)

    validation_gt = data_gen.flow_from_directory(path_validation,
                                                 target_size=tgt_size,
                                                 color_mode='rgb',
                                                 batch_size=32,
                                                 class_mode='categorical',
                                                 shuffle=False)

    return train_gt, test_gt, validation_gt


save_folder_path = "C:/Users/ItamarGIP/PycharmProjects/Banana-Learning/saved_models"


def save_model(model, model_name):
    # path_model = save_folder_path
    # save_dir = os.path.join(os.getcwd(), ) + path_model
    save_dir = save_folder_path
    # Save model and weights
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    model_path = os.path.join(save_dir, model_name + ".h5")
    model.save(model_path)
    print('Saved trained model at %s ' % model_path)


def execute(model, model_name, use_augmentation):
    print("============== " + model_name + " Based network ==============")

    batch_size = 32
    epochs = 1

    steps_per_epoch = train_size / batch_size
    validation_steps = test_size / batch_size

    train_gt, test_gt, validation_gt = gen_iterators(tgt_size, use_augmentation)

    model.fit_generator(train_gt, steps_per_epoch=steps_per_epoch,
                        epochs=epochs,
                        validation_data=test_gt,
                        validation_steps=validation_steps,
                        workers=4)

    scores = model.evaluate(validation_gt, verbose=1)
    print('Test loss:', scores[0])
    print('Test accuracy:', scores[1])

    save_model(model, model_name)


# ----------------- defines -----------------
tgt_size = (336, 252)
train_size = count_files_in_path(path_train)
test_size = count_files_in_path(path_test)
validation_size = count_files_in_path(path_validation)


# ----------------- models -----------------

# based on CIFAR-10 CNN structure
def define_cnn_model(opt, shape):
    num_classes = 4
    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same',
                     input_shape=shape))
    model.add(Activation('relu'))
    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    # model.add(Flatten())
    model.add(GlobalAveragePooling2D())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])
    return model


# # based on previous project
def define_cnn_model_prev(opt, shape):
    num_classes = 4
    model = Sequential()
    model.add(Conv2D(16, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='sigmoid'))

    model.compile(loss='categorical_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])

    return model


# ----------------- driver-code -----------------


cnn_model = define_cnn_model("ADAM", tgt_size + (3,))
prev_cnn_model = define_cnn_model_prev("ADAM", tgt_size + (3,))


train_gt, test_gt, validation_gt = gen_iterators(tgt_size, False)
print("========= CIFAR-10 MODEL - NO AUGMENTATION =========")
execute(cnn_model, "CIFAR-10-NO-AUG", False)
CIFAR10_PATH = "C:/Users/ItamarGIP/PycharmProjects/Banana-Learning/saved_models/CIFAR-10-NO-AUG.h5"
model = load_model(CIFAR10_PATH)
scores = model.evaluate(validation_gt, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])

print("========= CIFAR-10 MODEL - WITH AUGMENTATION =========")
execute(cnn_model, "CIFAR-10-YES-AUG", True)
CIFAR10_PATH = "C:/Users/ItamarGIP/PycharmProjects/Banana-Learning/saved_models/CIFAR-10-YES-AUG.h5"
model = load_model(CIFAR10_PATH)
scores = model.evaluate(validation_gt, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])


# predict = model.predict_generator(validation_gt) #need to shuffle_off
#
# for i in validation_gt:
#     idx = (validation_gt.batch_index - 1) * validation_gt.batch_size
#     print(validation_gt.filenames[idx : idx + validation_gt.batch_size])


# #y_pred = np.rint(predict)
# y_true = validation_gt.classes
#
# print(y_true)

