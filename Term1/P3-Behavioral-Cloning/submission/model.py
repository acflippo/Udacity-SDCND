import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import pandas as pd

from PIL import Image
import glob
import cv2
import random
from pathlib import Path

from PIL import Image
import keras
import math
import random
from sklearn.model_selection import train_test_split
from keras.optimizers import SGD, Adam, RMSprop
from scipy.misc.pilutil import imresize
"""
Steering angle prediction model
"""
import os
import argparse
import json
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Lambda, ELU, MaxPooling2D, Activation, Dense
from keras.layers import Convolution2D, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator


######## Load Data ########
# Reading Udacity's data
df = pd.read_csv('driving_log_udacity.csv', header=0)
df.columns = ["center_image", "left_image", "right_image", "steering_angle", "throttle", "break", "speed"]

# Reading My Recovery data
df_recovery = pd.read_csv('driving_log_recovery.csv', header=0)
df_recovery.columns = ["center_image", "left_image", "right_image", "steering_angle", "throttle", "break", "speed"]

# This is a 160 pixel x 320 pixel x 3 channels
img_center = mpimg.imread(df["center_image"][0].strip())
IMG_HEIGHT, IMG_WIDTH, IMG_CHANNEL = img_center.shape
print("img_height:", IMG_HEIGHT)
print("img_width:", IMG_WIDTH)
print("img_channel:", IMG_CHANNEL)

###### Up-sampling large turns and Down-sampling 0 degree angle Data for Training #######

###### Processing Udacity's Data ########
df_right = []
df_left = []
df_center = []
for i in range(len(df)):
    center_img = df["center_image"][i]
    left_img = df["left_image"][i]
    right_img = df["right_image"][i]
    angle = df["steering_angle"][i]
    
    if (angle > 0.15):
        df_right.append([center_img, left_img, right_img, angle])

        # I'm adding a small deviation of the angle 
        # This is to create more right turning samples for the same image
        for i in range(10):
            new_angle = angle * (1.0 + np.random.uniform(-1, 1)/30.0)
            df_right.append([center_img, left_img, right_img, angle])

    elif (angle < -0.15):
        df_left.append([center_img, left_img, right_img, angle])

        # I'm adding a small deviation of the angle 
        # This is to create more left turning samples for the same image
        for i in range(15):
            new_angle = angle * (1.0 + np.random.uniform(-1, 1)/30.0)
            df_left.append([center_img, left_img, right_img, new_angle])

    else:
        if (angle != 0):
            # Include all near 0 angle data but not exactly 0 degree angle
            df_center.append([center_img, left_img, right_img, angle])


###### Processing Recovery's Data ########
for i in range(len(df_recovery)):
    center_img = df_recovery["center_image"][i]
    left_img = df_recovery["left_image"][i]
    right_img = df_recovery["right_image"][i]
    angle = df_recovery["steering_angle"][i]
    
    if (angle > 0.15):
        df_right.append([center_img, left_img, right_img, angle])
        
        # I'm adding a small deviation of the angle 
        # This is to create more right turning samples for the same image
        for i in range(20):
            new_angle = angle * (1.0 + np.random.uniform(-1, 1)/30.0)
            df_right.append([center_img, left_img, right_img, angle])
            
    elif (angle < -0.15):
        df_left.append([center_img, left_img, right_img, angle])
        
        # I'm adding a small deviation of the angle
        # This is to create more left turning samples for the same image
        for i in range(10):
            new_angle = angle * (1.0 + np.random.uniform(-1, 1)/30.0)
            df_left.append([center_img, left_img, right_img, new_angle])
            
    else:
        for i in range(5):
            new_angle = angle * (1.0 + np.random.uniform(-1, 1)/30.0)
            df_center.append([center_img, left_img, right_img, new_angle])


# Shuffle the data so they're no longer sequential in the order that the data was collected 
random.shuffle(df_center)
random.shuffle(df_left)
random.shuffle(df_right)

df_center = pd.DataFrame(df_center, columns=["center_image", "left_image", "right_image", "steering_angle"])
df_left = pd.DataFrame(df_left, columns=["center_image", "left_image", "right_image", "steering_angle"])
df_right = pd.DataFrame(df_right, columns=["center_image", "left_image", "right_image", "steering_angle"])

# Put the Left, Right & Center together
data_list = [df_center, df_left, df_right]
data_list_df = pd.concat(data_list, ignore_index=True)

# y_data is not used as it is already contained in X_data's column
# but I made it to conform to the train_test_split function format
X_data = data_list_df[["center_image","left_image","right_image","steering_angle"]]
y_data = data_list_df["steering_angle"]

X_data = pd.DataFrame(X_data, columns=["center_image", "left_image", "right_image", "steering_angle"])
y_data = pd.DataFrame(y_data, columns=["steering_angle"])

######## Split into Training and Validation Data ########
X_train_data, X_valid_data, y_train_data, y_valid_data = train_test_split(X_data, y_data, test_size=0.2)

####### Reset Index on DataFrame ########
X_train_data = X_train_data.reset_index(drop=True)
X_valid_data = X_valid_data.reset_index(drop=True)

####################################################################

def change_brightness(image):
    # Randomly select a percent change
    change_pct = random.uniform(0.4, 1.2)
    
    # Change to HSV to change the brightness V
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    hsv[:,:,2] = hsv[:,:,2] * change_pct
    
    #Convert back to RGB 
    img_bright = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    return img_bright

####################################################################

def flip_image(image, angle):
    img_flip = cv2.flip(image,1)
    angle = -angle
        
    return img_flip, angle

####################################################################

def preprocessImage(image):
    # Proportionally get lower half portion of the image
    nrow, ncol, nchannel = image.shape
    
    start_row = int(nrow * 0.35)
    end_row = int(nrow * 0.875)   
    
    ## This removes most of the sky and small amount below including the hood
    image_no_sky = image[start_row:end_row, :]

    # This resizes to 66 x 220 for NVIDIA's model
    new_image = cv2.resize(image_no_sky, (220,66), interpolation=cv2.INTER_AREA)

    return new_image

########################################

def preprocess_image_train(data_row_df):

    path_filename = data_row_df["center_image"][0]
    image = cv2.imread(path_filename)
    angle = data_row_df['steering_angle'][0]
    
    image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    image = change_brightness(image)

    # Coin flip to see to flip image and create a new sample of -angle
    if np.random.randint(2) == 1:
        image, angle = flip_image(image, angle)
    
    # This preprocessImage() needs to be done in drive.py
    image = preprocessImage(image)
    image = np.array(image)

    return image, angle

########################################

def preprocess_image_valid(data_row_df):

    path_filename = data_row_df["center_image"][0]
    angle = data_row_df['steering_angle'][0]
    image = cv2.imread(path_filename)
    
    image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    
    # This preprocessImage() needs to be done in drive.py
    image = preprocessImage(image)
    image = np.array(image)
    
    return image, angle

########################################

# NVIDIA's input parameters
INPUT_IMG_HEIGHT = 66
INPUT_IMG_WIDTH = 220

# This will generate a batch of new images of final input dimensions 
# and the y output (steering_angles)
def generate_batch_train_from_dataframe(data_df, batch_size = 128):
    
    batch_images = np.zeros((batch_size, INPUT_IMG_HEIGHT, INPUT_IMG_WIDTH, 3))
    batch_angles = np.zeros(batch_size)
    
    while True:
        for i in range (batch_size):
            # Randomly get a sample from the input data
            idx = np.random.randint(len(data_df))

            # reset_index sets this data_df starting row to 0
            data_row = data_df.iloc[[idx]].reset_index()
            img1, angle1 = preprocess_image_train(data_row)

            batch_images[i] = img1
            batch_angles[i] = angle1

        yield batch_images, batch_angles

########################################
    
def generate_valid_from_dataframe(data_df):
    while True:
        for idx in range(len(data_df)):
            data_row = data_df.iloc[[idx]].reset_index()
            img, angle = preprocess_image_valid(data_row)

            # Since not stacking the images, it's shape remains (height, width, channel)
            # but need it to be (1, height, width, channel)
            img = img.reshape(1, img.shape[0], img.shape[1], img.shape[2])
            angle = np.array([[angle]])
            yield img, angle

########################################
# Initialize generator
valid_data_generator = generate_valid_from_dataframe(X_valid_data)

########################################

def save_model(fileModelJSON, fileWeights):
    prefix = "model/"
    
    filenameJSON = prefix + fileModelJSON
    if Path(filenameJSON).is_file():
        os.remove(filenameJSON)    
    with open (filenameJSON, 'w') as outfile:
        json.dump(model.to_json(), outfile)
        
    filenameWeights = prefix + fileWeights
    if Path(filenameWeights).is_file():
        os.remove(filenameWeights)
    model.save_weights(filenameWeights, True)

########################################
# Parameters for training
input_shape = (INPUT_IMG_HEIGHT, INPUT_IMG_WIDTH, 3)

########################################
def get_model_nvidia():
    model = Sequential()

    model.add(Lambda(lambda x: x/127.5 - 1.,input_shape = input_shape))
    model.add(Convolution2D(24, 5, 5, subsample=(2, 2), border_mode="valid", init='he_normal', name='conv1'))
    model.add(ELU())
    model.add(Convolution2D(36, 5, 5, subsample=(2, 2), border_mode="valid", init='he_normal', name='conv2'))
    model.add(ELU())
    model.add(Convolution2D(48, 5, 5, subsample=(2, 2), border_mode="valid", init='he_normal', name='conv3'))
    model.add(ELU())
    model.add(Convolution2D(64, 3, 3, subsample=(1, 1), border_mode="valid", init='he_normal', name='conv4'))
    model.add(ELU())
    model.add(Convolution2D(64, 3, 3, subsample=(1, 1), border_mode="valid", init='he_normal', name='conv5'))
    model.add(Flatten(name='flatten1'))
    model.add(ELU())
    model.add(Dense(1164, init='he_normal', name='dense1'))
    model.add(ELU())
    model.add(Dense(100, init='he_normal', name='dense2'))
    model.add(ELU())
    model.add(Dense(50, init='he_normal', name='dense3'))
    model.add(ELU())
    model.add(Dense(10, init='he_normal', name='dense4'))
    model.add(ELU())
    model.add(Dense(1, init='he_normal', name='dense5'))

    adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.01)
    model.compile(optimizer=adam, loss='mse')
    
    return model

########################################
# Start the training and save models 
val_size = len(X_valid_data)
batch_size = 512

idx_best = 0
val_best = 9999

for idx in range(3):
    print(">>> Iteration :", idx, " <<<")
    train_data_generator = generate_batch_train_from_dataframe(X_train_data, batch_size)
    
    #model = get_model_keras_tutorial()
    #model = get_model_comma_ai()
    model = get_model_nvidia()
    history = model.fit_generator(train_data_generator, samples_per_epoch=20480,
                                 nb_epoch=6, validation_data=valid_data_generator,
                                 nb_val_samples=val_size)
    
    fileModelJSON = 'model_nvidia_v2' + str(idx) + '.json'
    fileWeights = 'model_nvidia_v2' + str(idx) + '.h5'
    save_model(fileModelJSON, fileWeights)
    
    val_loss = history.history['val_loss'][0]
    
    # If found a small val_loss then that's the new val_best
    if val_loss < val_best:
        val_best = val_loss
        idx_best = idx

print("Best model found at idx:", idx_best)
print("Best Validation score:", val_best)
