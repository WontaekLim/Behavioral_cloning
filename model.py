# Behavioral clonning
# Udacity project 31x98
#
# Author: Wonteak Lim
# Description
# This code is a neural network for end-to-end driving
# And it is based on the nconda's implementation (https://github.com/ncondo/CarND-Behavioral-Cloning)

import tensorflow as tf
import numpy as np
import random
import csv
import cv2 
import json
import h5py

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from keras.layers import Activation, Dense, Dropout, ELU, Flatten, Input, Lambda
from keras.layers.convolutional import Convolution2D, Cropping2D
from keras.models import Sequential, Model, load_model, model_from_json
from keras.regularizers import l2


def get_data(log_file):
    image_names, steering_angles = [], []    
    steering_offset = 0.275
    with open(log_file, 'r') as f:
        reader = csv.reader(f)
        next(reader, None)
        for center_img, left_img, right_img, angle, _, _, _ in reader:
            angle = float(angle)
            image_names.append([center_img.strip(), left_img.strip(), right_img.strip()])
            steering_angles.append([angle, angle+steering_offset, angle-steering_offset])

    return image_names, steering_angles


def generate_batch(X_train, y_train, batch_size=64):
    images = np.zeros((batch_size, 66, 200, 3), dtype=np.float32)
    angles = np.zeros((batch_size,), dtype=np.float32)
    while True:
        straight_count = 0
        for i in range(batch_size):
            # Select a random index to use for data sample
            sample_index = random.randrange(len(X_train))
            image_index = random.randrange(len(X_train[0]))
            angle = y_train[sample_index][image_index]
            # Limit angles of less than absolute value of .1 to no more than 1/2 of data
            # to reduce bias of car driving straight
            if abs(angle) < .1:
                straight_count += 1
            if straight_count > (batch_size * .5):
                while abs(y_train[sample_index][image_index]) < .1:
                    sample_index = random.randrange(len(X_train))
            # Read image in from directory, process, and convert to numpy array
            image = cv2.imread('data/' + str(X_train[sample_index][image_index]))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = process_image(image)
            image = np.array(image, dtype=np.float32)
            # Flip image and apply opposite angle 50% of the time
            if random.randrange(2) == 1:
                image = cv2.flip(image, 1)
                angle = -angle
            images[i] = image
            angles[i] = angle
        yield images, angles


def resize(image):
    return cv2.resize(image, (200, 66), interpolation=cv2.INTER_AREA)


def normalize(image):
    return image / 127.5 - 1.


def crop_image(image):
    return image[40:-20,:]


def random_brightness(image):
    image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    brightness = .25 + np.random.uniform()
    image[:,:,2] = image[:,:,2] * brightness
    image = cv2.cvtColor(image, cv2.COLOR_HSV2RGB)
    return image


def process_image(image):
    image = random_brightness(image)
    image = crop_image(image)
    image = resize(image)
    return image


def create_model():
    model = Sequential([
        # Normalize image to -1.0 to 1.0
        Lambda(normalize, input_shape=(66, 200, 3)),
        # Convolutional layer 1 24@31x98 | 5x5 kernel | 2x2 stride | elu activation 
        Convolution2D(24, 5, 5, border_mode='valid', activation='elu', subsample=(2, 2), init='he_normal', W_regularizer=l2(0.001)),
        # Dropout with drop probability of .1 (keep probability of .9)
        Dropout(.3),
        # Convolutional layer 2 36@14x47 | 5x5 kernel | 2x2 stride | elu activation
        Convolution2D(36, 5, 5, border_mode='valid', activation='elu', subsample=(2, 2), init='he_normal', W_regularizer=l2(0.001)),
        # Dropout with drop probability of .2 (keep probability of .8)
        Dropout(.3),
        # Convolutional layer 3 48@5x22  | 5x5 kernel | 2x2 stride | elu activation
        Convolution2D(48, 5, 5, border_mode='valid', activation='elu', subsample=(2, 2), init='he_normal', W_regularizer=l2(0.001)),
        # Dropout with drop probability of .2 (keep probability of .8)
        Dropout(.3),
        # Convolutional layer 4 64@3x20  | 3x3 kernel | 1x1 stride | elu activation
        Convolution2D(64, 3, 3, border_mode='valid', activation='elu', subsample=(1, 1), init='he_normal', W_regularizer=l2(0.001)),
        # Dropout with drop probability of .2 (keep probability of .8)
        Dropout(.3),
        # Convolutional layer 5 64@1x18  | 3x3 kernel | 1x1 stride | elu activation
        Convolution2D(64, 3, 3, border_mode='valid', activation='elu', subsample=(1, 1), init='he_normal', W_regularizer=l2(0.001)),
        # Flatten
        Flatten(),
        # Dropout with drop probability of .3 (keep probability of .7)
        Dropout(.3),
        # Fully-connected layer 1 | 100 neurons | elu activation
        Dense(100, activation='elu', init='he_normal', W_regularizer=l2(0.001)),
        # Dropout with drop probability of .5
        Dropout(.3),
        # Fully-connected layer 2 | 50 neurons | elu activation
        Dense(50, activation='elu', init='he_normal', W_regularizer=l2(0.001)),
        # Dropout with drop probability of .5
        Dropout(.3),
        # Fully-connected layer 3 | 10 neurons | elu activation
        Dense(10, activation='elu', init='he_normal', W_regularizer=l2(0.001)),
        # Dropout with drop probability of .5
        Dropout(.3),
        # Output
        Dense(1, activation='linear', init='he_normal')
    ])

    model.compile(optimizer='adam', loss='mse')
    return model    


if __name__=="__main__":

    # Get the training data
    X_train, y_train = get_data('data/driving_log.csv')
    X_train, y_train = shuffle(X_train, y_train, random_state=14)
    X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.1, random_state=14)

    # generate model
    model = create_model()    
    model.fit_generator(generate_batch(X_train, y_train), samples_per_epoch=24000, nb_epoch=12, validation_data=generate_batch(X_validation, y_validation), nb_val_samples=1024)

    print('Saving model weights and configuration file.')
    # Save model weights
    model.save_weights('model.h5')
    # Save model architecture as json file
    with open('model.json', 'w') as outfile:
        json.dump(model.to_json(), outfile)

    # Explicitly end tensorflow session
    from keras import backend as K 

    K.clear_session()
