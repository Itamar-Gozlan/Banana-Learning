from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pylab as plt

data_path = "D:/Users Data/ItamarGIP/Desktop/Itamar/data/sorted"

# datagen = ImageDataGenerator()
# # prepare an iterators for each dataset
# train_it = datagen.flow_from_directory(data_path + '/train/')
# val_it = datagen.flow_from_directory(data_path + '/validation/')
# test_it = datagen.flow_from_directory(data_path + '/test/')
#
# # confirm the iterator works
# batchX, batchy = train_it.next()
# print('Batch shape=%s, min=%.3f, max=%.3f' % (batchX.shape, batchX.min(), batchX.max()))

im = plt.imread("D:/Users Data/ItamarGIP/Desktop/Itamar/data/sorted/train/treat_a/20180911_153138_RGB_Treat_A_01.jpg")

## Done:
# - sort folder and disterbute to different folders
# - load images
## TODO
#  - load all images
#  - train network rouphly (proof of concept)
# - image processing (remove background!)