from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
import tensorflow as tf
import os
import math


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


gpu_options = tf.compat.v1.GPUOptions(allow_growth=True)
session = tf.compat.v1.InteractiveSession(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))

# gpus = tf.config.experimental.list_physical_devices('GPU') 
# for gpu in gpus: 
#         tf.config.experimental.set_memory_growth(gpu, True)

model = Sequential()
model.add(Convolution2D(32, (3, 3), activation='relu', input_shape=(64, 64, 1))) 
model.add(MaxPooling2D((2, 2)))
model.add(Convolution2D(64, (3, 3), activation='relu')) 
model.add(MaxPooling2D((2, 2)))
model.add(Convolution2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(3, activation='softmax'))


# 1. Building CNN

# classifier = Sequential()

# # First convolution layer and pooling
# classifier.add(Convolution2D(32, (3, 3), input_shape=(64, 64, 1), activation='relu'))
# classifier.add(MaxPooling2D(pool_size=(2, 2)))
# # Second convolution layer and pooling
# classifier.add(Convolution2D(32, (3, 3), activation='relu'))
# # input_shape is going to be the pooled feature maps from the previous convolution layer
# classifier.add(MaxPooling2D(pool_size=(2, 2)))

# # Flattening the layers
# classifier.add(Flatten())

# # Adding a fully connected layer
# classifier.add(Dense(units=128, activation='relu'))
# classifier.add(Dense(units=8, activation='softmax')) # softmax for more than 2

# Compiling the CNN
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy']) # categorical_crossentropy for more than 2

# Step 2 - Preparing the train/test data and training the model

# Code copied from - https://keras.io/preprocessing/image/

 

image_gen = ImageDataGenerator(rotation_range=30, # rotate the image 30 degrees
                               width_shift_range=0.1, # Shift the pic width by a max of 10%
                               height_shift_range=0.1, # Shift the pic height by a max of 10%
                               rescale=1./255, # Rescale the image by normalzing it.
                               shear_range=0.2, # Shear means cutting away part of the image (max 20%)
                               zoom_range=0.0, # Zoom in by 20% max
                               horizontal_flip=True, # Allo horizontal flipping
                               fill_mode='nearest' # Fill in missing pixels with the nearest filled value
                              )

training_set = image_gen.flow_from_directory('data/train',
                                                 target_size=(64, 64),
                                                 batch_size=30,
                                                 color_mode='grayscale',
                                                 class_mode='binary')

test_set = image_gen.flow_from_directory('data/test',
                                            target_size=(64, 64),
                                            batch_size=3 ,
                                            color_mode='grayscale',
                                            class_mode='binary')                              

# dataset = tf.data.Dataset.from_tensor_slices((images, targets)) \
#         .batch(12, drop_remainder=True)

model.fit(
        training_set,
        steps_per_epoch = 6 , #len(test_set)//BATCH_SIZE, # No of images in training set
        epochs= 50,
        validation_data=test_set,
        validation_steps = 6 ) #len(training_set)//BATCH_SIZE )#


# print(training_set.class_indices)

# Saving the model
model_json = model.to_json()
with open("model-bw.json", "w") as json_file:
    json_file.write(model_json)
model.save_weights('model-bw.h5')
print("Done")








# < TEST 4 > #

# from keras.models import Sequential
# from keras.layers import Convolution2D
# from keras.layers import MaxPooling2D
# from keras.layers import Flatten
# from keras.layers import Dense
# import tensorflow as tf
# import os
# import math
# from keras.preprocessing.image import ImageDataGenerator



# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


# gpu_options = tf.compat.v1.GPUOptions(allow_growth=True)
# session = tf.compat.v1.InteractiveSession(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))

# # gpus = tf.config.experimental.list_physical_devices('GPU') 
# # for gpu in gpus: 
# #         tf.config.experimental.set_memory_growth(gpu, True)

# # 1. Building CNN

# classifier = Sequential()

# # First convolution layer and pooling
# classifier.add(Convolution2D(32, (3, 3), input_shape=(64, 64, 1), activation='relu'))
# classifier.add(MaxPooling2D(pool_size=(2, 2)))
# # Second convolution layer and pooling
# classifier.add(Convolution2D(32, (3, 3), activation='relu'))
# # input_shape is going to be the pooled feature maps from the previous convolution layer
# classifier.add(MaxPooling2D(pool_size=(2, 2)))

# # Flattening the layers
# classifier.add(Flatten())

# # Adding a fully connected layer
# classifier.add(Dense(units=128, activation='relu'))
# classifier.add(Dense(units=8, activation='softmax')) # softmax for more than 2

# # Compiling the CNN
# classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy']) # categorical_crossentropy for more than 2

# # Step 2 - Preparing the train/test data and training the model

# # Code copied from - https://keras.io/preprocessing/image/

 

# image_gen = ImageDataGenerator(rotation_range=30, # rotate the image 30 degrees
#                                width_shift_range=0.1, # Shift the pic width by a max of 10%
#                                height_shift_range=0.1, # Shift the pic height by a max of 10%
#                                rescale=1./255, # Rescale the image by normalzing it.
#                                shear_range=0.2, # Shear means cutting away part of the image (max 20%)
#                                zoom_range=0.2, # Zoom in by 20% max
#                                horizontal_flip=True, # Allo horizontal flipping
#                                fill_mode='nearest' # Fill in missing pixels with the nearest filled value
#                               )

# training_set = image_gen.flow_from_directory('data/train',
#                                                  target_size=(64, 64),
#                                                  batch_size=100,
#                                                  color_mode='grayscale',
#                                                  class_mode='categorical')

# test_set = image_gen.flow_from_directory('data/test',
#                                             target_size=(64, 64),
#                                             batch_size=10 ,
#                                             color_mode='grayscale',
#                                             class_mode='categorical')                              

# # dataset = tf.data.Dataset.from_tensor_slices((images, targets)) \
# #         .batch(12, drop_remainder=True)

# classifier.fit(
#         training_set,
#         steps_per_epoch = 7 , #len(test_set)//BATCH_SIZE, # No of images in training set
#         epochs= 50,
#         validation_data=test_set,
#         validation_steps = 7 ) #len(training_set)//BATCH_SIZE )#


# # print(training_set.class_indices)

# # Saving the model
# model_json = classifier.to_json()
# with open("model-bw.json", "w") as json_file:
#     json_file.write(model_json)
# classifier.save_weights('model-bw.h5')








# < TEST 3 > #

# from keras.models import Sequential
# from keras.layers import Convolution2D
# from keras.layers import MaxPooling2D
# from keras.layers import Flatten
# from keras.layers import Dense
# import tensorflow as tf


# from keras.preprocessing.image import ImageDataGenerator

# gpu_options = tf.compat.v1.GPUOptions(allow_growth=True)
# session = tf.compat.v1.InteractiveSession(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))

# # gpus = tf.config.experimental.list_physical_devices('GPU') 
# # for gpu in gpus: 
# #         tf.config.experimental.set_memory_growth(gpu, True)

# # 1. Building CNN

# classifier = Sequential()

# # First convolution layer and pooling
# classifier.add(Convolution2D(32, (3, 3), input_shape=(64, 64, 1), activation='relu'))
# classifier.add(MaxPooling2D(pool_size=(2, 2)))
# # Second convolution layer and pooling
# classifier.add(Convolution2D(32, (3, 3), activation='relu'))
# # input_shape is going to be the pooled feature maps from the previous convolution layer
# classifier.add(MaxPooling2D(pool_size=(2, 2)))

# # Flattening the layers
# classifier.add(Flatten())

# # Adding a fully connected layer
# classifier.add(Dense(units=128, activation='relu'))
# classifier.add(Dense(units=8, activation='softmax')) # softmax for more than 2

# # Compiling the CNN
# classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy']) # categorical_crossentropy for more than 2

# # Step 2 - Preparing the train/test data and training the model

# # Code copied from - https://keras.io/preprocessing/image/

 

# image_gen = ImageDataGenerator(rotation_range=30, # rotate the image 30 degrees
#                                width_shift_range=0.1, # Shift the pic width by a max of 10%
#                                height_shift_range=0.1, # Shift the pic height by a max of 10%
#                                rescale=1./255, # Rescale the image by normalzing it.
#                                shear_range=0.2, # Shear means cutting away part of the image (max 20%)
#                                zoom_range=0.2, # Zoom in by 20% max
#                                horizontal_flip=True, # Allo horizontal flipping
#                                fill_mode='nearest' # Fill in missing pixels with the nearest filled value
#                               )

# training_set = image_gen.flow_from_directory('data/train',
#                                                  target_size=(64, 64),
#                                                  batch_size=91980,
#                                                  color_mode='grayscale',
#                                                  class_mode='categorical')

# test_set = image_gen.flow_from_directory('data/test',
#                                             target_size=(64, 64),
#                                             batch_size=4920,
#                                             color_mode='grayscale',
#                                             class_mode='categorical')                              


# classifier.fit_generator(
#         training_set,
#         steps_per_epoch= 9198, # No of images in training set
#         epochs= 10,
#         validation_data=test_set,
#         validation_steps=492 )#

# # print(training_set.class_indices)

# # Saving the model
# model_json = classifier.to_json()
# with open("model-bw.json", "w") as json_file:
#     json_file.write(model_json)
# classifier.save_weights('model-bw.h5')

















# < TEST 2 > #

# from tensorflow.python.keras.engine.sequential import Sequential
# from PIL.Image import Image
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
# from tensorflow.keras.preprocessing import image 
# from tensorflow.keras.optimizers import RMSprop
# import matplotlib.pyplot as plt
# import tensorflow as tf
# import cv2 as cv
# import os

# train_datagen = ImageDataGenerator(
#         rescale=1./255,
#         shear_range=0.2,
#         zoom_range=0.2,
#         horizontal_flip=True)

# test_datagen = ImageDataGenerator(rescale=1./255)

# training_set = train_datagen.flow_from_directory('data/train',
#                                                  target_size=(64, 64),
#                                                  batch_size=30,
#                                                  color_mode='grayscale',
#                                                  class_mode='categorical')

# test_set = test_datagen.flow_from_directory('data/test',
#                                             target_size=(64, 64),
#                                             batch_size=30,
#                                             color_mode='grayscale',
#                                             class_mode='categorical') 

# # image_gen = ImageDataGenerator(rotation_range=30, # rotate the image 30 degrees
# #                                width_shift_range=0.1, # Shift the pic width by a max of 10%
# #                                height_shift_range=0.1, # Shift the pic height by a max of 10%
# #                                rescale=1/255, # Rescale the image by normalzing it.
# #                                shear_range=0.2, # Shear means cutting away part of the image (max 20%)
# #                                zoom_range=0.2, # Zoom in by 20% max
# #                                horizontal_flip=True, # Allo horizontal flipping
# #                                fill_mode='nearest' # Fill in missing pixels with the nearest filled value
# #                               )
# # training_set = image_gen.flow_from_directory('data/train')
# # test_set = image_gen.flow_from_directory('data/test')


# # print(training_set.classes)

# model = tf.keras.models.Sequential([

#         tf.keras.layers.Conv2D(32,(3,3),activation='relu',input_shape = (64,64,1)),
#         tf.keras.layers.MaxPool2D(2,2),

#         tf.keras.layers.Conv2D(32,(3,3),activation='relu'),
#         tf.keras.layers.MaxPool2D(2,2),

#         tf.keras.layers.Conv2D(32,(3,3),activation='relu'),
#         tf.keras.layers.MaxPool2D(2,2),
        
#         tf.keras.layers.Flatten(),

#         tf.keras.layers.Dense(units = 128,activation= 'relu'),

#         tf.keras.layers.Dense(units= 8,activation= 'softmax')
# ])

# model.compile(optimizer='adam', 
#                 loss='categorical_crossentropy', 
#                 metrics=['accuracy']
# )

# model_fit = model.fit(
#         training_set,
#         steps_per_epoch= 9198, # No of images in training set
#         epochs= 10,
#         validation_data=test_set,
#         validation_steps=492 # No of images in test set
# )

# # # Saving the model
# model_json = model.to_json()
# with open("model-bw.json", "w") as json_file:
#     json_file.write(model_json)
# model.save_weights('model-bw.h5')
# print("She's ready!")













# < TEST 1 > #

# # Importing the Keras libraries and packages
# from keras.models import Sequential
# from keras.layers import Convolution2D
# from keras.layers import MaxPooling2D
# from keras.layers import Flatten
# from keras.layers import Dense
# from keras.preprocessing.image import ImageDataGenerator

# # Step 1 - Building the CNN

# # Initializing the CNN
# classifier = Sequential()

# # First convolution layer and pooling
# classifier.add(Convolution2D(32, (3, 3), input_shape=(64, 64, 1), activation='relu'))
# classifier.add(MaxPooling2D(pool_size=(2, 2)))
# # Second convolution layer and pooling
# classifier.add(Convolution2D(32, (3, 3), activation='relu'))
# # input_shape is going to be the pooled feature maps from the previous convolution layer
# classifier.add(MaxPooling2D(pool_size=(2, 2)))

# # Flattening the layers
# classifier.add(Flatten())

# # Adding a fully connected layer
# classifier.add(Dense(units=128, activation='relu'))
# classifier.add(Dense(units=6, activation='softmax')) # softmax for more than 2

# # Compiling the CNN
# classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy']) # categorical_crossentropy for more than 2


# # Step 2 - Preparing the train/test data and training the model

# # Code copied from - https://keras.io/preprocessing/image/

# train_datagen = ImageDataGenerator(
#         rescale=1./255,
#         shear_range=0.2,
#         zoom_range=0.2,
#         horizontal_flip=True)

# test_datagen = ImageDataGenerator(rescale=1./255)

# training_set = train_datagen.flow_from_directory('data/train',
#                                                  target_size=(64, 64),
#                                                  batch_size=5,
#                                                  color_mode='grayscale',
#                                                  class_mode='categorical')

# test_set = test_datagen.flow_from_directory('data/test',
#                                             target_size=(64, 64),
#                                             batch_size=5,
#                                             color_mode='grayscale',
#                                             class_mode='categorical') 

# image_gen = ImageDataGenerator(rotation_range=30, # rotate the image 30 degrees
#                                width_shift_range=0.1, # Shift the pic width by a max of 10%
#                                height_shift_range=0.1, # Shift the pic height by a max of 10%
#                                rescale=1/255, # Rescale the image by normalzing it.
#                                shear_range=0.2, # Shear means cutting away part of the image (max 20%)
#                                zoom_range=0.2, # Zoom in by 20% max
#                                horizontal_flip=True, # Allo horizontal flipping
#                                fill_mode='nearest' # Fill in missing pixels with the nearest filled value
#                               )

# image_gen.flow_from_directory('data/train/palm')
# image_gen.flow_from_directory('data/train/twoF')
# image_gen.flow_from_directory('data/train/fiveF')
# image_gen.flow_from_directory('data/train/broFist')
# image_gen.flow_from_directory('data/train/Lshape')
# image_gen.flow_from_directory('data/train/okay')


# classifier.fit_generator(
#         training_set,
#         steps_per_epoch= 500, # No of images in training set
#         epochs= 10,
#         validation_data=test_set,
#         validation_steps=30)# No of images in test set

# # print(training_set.class_indices)

# # Saving the model
# model_json = classifier.to_json()
# with open("model-bw.json", "w") as json_file:
#     json_file.write(model_json)
# classifier.save_weights('model-bw.h5')
# print("Your model is trained!")
