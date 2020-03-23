from __future__ import print_function
import sys
import numpy as np
import matplotlib.pylab as plt
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, GlobalAveragePooling2D
from keras.layers import Conv2D, MaxPooling2D
import os
import traceback

####### should be imported from utils #########



def populate_path_list(src_dir):
    path_list = []
    for dir_path, dirs, files in os.walk(src_dir, topdown=False):
        for name in files:
            path_list.append(os.path.join(dir_path, name))
    return path_list


data_purpose = ["train", "test", "validation"]
categories = ["treat_a", "treat_b", "treat_c", "treat_d"]

data_path = "D:/Users Data/ItamarGIP/Desktop/Itamar/data/sorted"
#################################################


def load_per_data_purpose(data_root_path, shape_in, purpose, verbose=False):
    entry = (0,) + shape_in
    x_tgt = np.empty(entry, dtype='uint8')
    y_tgt = np.empty((0,), dtype='uint8')
    if verbose: print("Loading per data purpose: ",purpose)
    for ct in range(0,len(categories)):
        path_list = populate_path_list(data_root_path + "/" + purpose + "/" + categories[ct])
        n_images = len(path_list)
        # n_images = 5 # TODO - test only!
        curr_entry = (n_images,) + shape_in
        curr_x = np.empty(curr_entry, dtype='uint8')
        if verbose: print("Loading images from category ",categories[ct])
        for img in range(0, n_images):
            if verbose:
                print(str(img) + "\t", end='')
                if img != 0 and (img % 25) == 0: print("")
            try:
                curr_x[img] = np.resize(plt.imread(path_list[img]), shape_in)
            except ValueError:
                # traceback.print_exc()
                print("ValueError: Error image path: " + path_list[img], file=sys.stderr)
        curr_y = np.empty((n_images), dtype='uint8')
        curr_y.fill(ct)
        y_tgt = np.append(y_tgt, curr_y)
        x_tgt = np.concatenate((x_tgt, curr_x), axis=0)
    return x_tgt, y_tgt


def load_images_to_array(data_root_path, shape_in, verbose=False):
    (x_train, y_train) = load_per_data_purpose(data_root_path, shape_in, "train", verbose)
    (x_test, y_test) = load_per_data_purpose(data_root_path, shape_in, "test", verbose)

    return  (x_train, y_train), (x_test, y_test)


############################ from CIFAR 10 example #########################################

batch_size = 32
num_classes = 4
epochs = 100
data_augmentation = False
num_predictions = 20
save_dir = os.path.join(os.getcwd(), 'banana_model')
model_name = 'bana_model_trained.h5'

# The data, split between train and test sets:
path_train = populate_path_list(data_path + "/train/treat_a")
im = plt.imread(path_train[0])
(x_train, y_train), (x_test, y_test) = load_images_to_array(data_path, (504,672,3), True)
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')
print('len(x_train) = ', len(x_train))
print('len(y_train) = ', len(y_train))
print('len(x_test) = ', len(x_test))
print('len(y_test) = ', len(y_test))

# Convert class vectors to binary class matrices.
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

model = Sequential()
model.add(Conv2D(32, (3, 3), padding='same',
                 input_shape=x_train.shape[1:]))
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

# initiate RMSprop optimizer
opt = keras.optimizers.RMSprop(lr=0.0001, decay=1e-6)

# Let's train the model using RMSprop
model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

x_train = x_train.astype('float16')
x_test = x_test.astype('float16')
x_train /= 255
x_test /= 255

if not data_augmentation:
    print('Not using data augmentation.')
    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              validation_data=(x_test, y_test),
              shuffle=True)
else:
    print('Using real-time data augmentation.')
    # This will do preprocessing and realtime data augmentation:
    datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        zca_epsilon=1e-06,  # epsilon for ZCA whitening
        rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
        # randomly shift images horizontally (fraction of total width)
        width_shift_range=0.1,
        # randomly shift images vertically (fraction of total height)
        height_shift_range=0.1,
        shear_range=0.,  # set range for random shear
        zoom_range=0.,  # set range for random zoom
        channel_shift_range=0.,  # set range for random channel shifts
        # set mode for filling points outside the input boundaries
        fill_mode='nearest',
        cval=0.,  # value used for fill_mode = "constant"
        horizontal_flip=True,  # randomly flip images
        vertical_flip=False,  # randomly flip images
        # set rescaling factor (applied before any other transformation)
        rescale=None,
        # set function that will be applied on each input
        preprocessing_function=None,
        # image data format, either "channels_first" or "channels_last"
        data_format=None,
        # fraction of images reserved for validation (strictly between 0 and 1)
        validation_split=0.0)

    # Compute quantities required for feature-wise normalization
    # (std, mean, and principal components if ZCA whitening is applied).
    datagen.fit(x_train)

    # Fit the model on the batches generated by datagen.flow().
    model.fit_generator(datagen.flow(x_train, y_train,
                                     batch_size=batch_size), steps_per_epoch = 10,
                        epochs=epochs,
                        validation_data=(x_test, y_test),
                        workers=4)

# Save model and weights
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
model_path = os.path.join(save_dir, model_name)
model.save(model_path)
print('Saved trained model at %s ' % model_path)

# Score trained model.
scores = model.evaluate(x_test, y_test, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])
