# Banana-Learning - Source Dir

## Folders
- current dir - experiments that define and execute convolutional networks.
- utils: data sorting and creating triplets from the dataset
- logs: previous experiments results and error logs

## Files
- native_cnn.py - contains two models
    - prev-CNN - was used in a previous project
    - CIFAR-10 - based on Keras CIFAR-10 architecture with small modifications (main network)
- transfer_learning.py - define 4 networks that is used with transfer learning
    - GoogleNet - main network
    - MobileNet - main network
    - VGG16
    - ResNet50
- train_main.bat - batch script that was used in anaconda prompt to run the desired experiment and auto-commit them to the repository

## How to use
- Order your data in the following structure:


```
root
├── Train
│   ├── a_class
│   ├── b_class
│   ├── c_class
│   └── d_class
├── Test
│   ├── a_class
│   ├── b_class
│   ├── c_class
│   └── d_class
└── Validation
    ├── a_class
    ├── b_class
    ├── c_class
    └── d_class
```


- Make sure that the validation group is completely separate from the train and test folders
- change the path (global variables) in for the experiment you wish to perform
- change attributes, make sure target size is compatible with the dataset
- edit the driver's code - please note that the driver's code committed is the last experiment I performed and needed to be changed
- more information in the comments inside the code

</br>
</br>
</br>
</br>
