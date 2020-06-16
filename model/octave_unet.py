# model code all in this cell

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import math
import numpy as np
import tensorflow as tf
from sklearn.utils import class_weight

from tensorflow.keras import layers
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import *
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras_octave_conv import OctaveConv2D
      
def unet(pretrained_weights = None,input_size = (800,600,1)):

    inputs = Input(input_size)
    # downsampling for low frequencies
    low = layers.AveragePooling2D(2)(inputs)

    high1, low1 = OctaveConv2D(64)([inputs,low])

    high1 = layers.BatchNormalization()(high1)
    high1 = layers.Activation("relu")(high1)
    low1 = layers.BatchNormalization()(low1)
    low1 = layers.Activation("relu")(low1)

    high1, low1 = OctaveConv2D(64)([high1, low1])

    high1 = layers.BatchNormalization()(high1)
    high1 = layers.Activation("relu")(high1)
    low1 = layers.BatchNormalization()(low1)
    low1 = layers.Activation("relu")(low1)

    pool1high = layers.MaxPooling2D(2)(high1)
    pool1low = layers.MaxPooling2D(2)(low1)
    
    high2, low2 = OctaveConv2D(128)([pool1high,pool1low])

    high2 = layers.BatchNormalization()(high2)
    high2 = layers.Activation("relu")(high2)
    low2 = layers.BatchNormalization()(low2)
    low2 = layers.Activation("relu")(low2)

    high2, low2 = OctaveConv2D(128)([high2, low2])

    high2 = layers.BatchNormalization()(high2)
    high2 = layers.Activation("relu")(high2)
    low2 = layers.BatchNormalization()(low2)
    low2 = layers.Activation("relu")(low2)

    pool2high = layers.MaxPooling2D(2)(high2)
    pool2low = layers.MaxPooling2D(2)(low2)
    
    high3, low3 = OctaveConv2D(256)([pool2high,pool2low])

    high3 = layers.BatchNormalization()(high3)
    high3 = layers.Activation("relu")(high3)
    low3 = layers.BatchNormalization()(low3)
    low3 = layers.Activation("relu")(low3)

    high3, low3 = OctaveConv2D(256)([high3, low3])

    high3 = layers.BatchNormalization()(high3)
    high3 = layers.Activation("relu")(high3)
    low3 = layers.BatchNormalization()(low3)
    low3 = layers.Activation("relu")(low3)

    pool3high = layers.MaxPooling2D(2)(high3)
    pool3low = layers.MaxPooling2D(2)(low3)
    
    high4, low4 = OctaveConv2D(512)([pool3high,pool3low])

    high4 = layers.BatchNormalization()(high4)
    high4 = layers.Activation("relu")(high4)
    low4 = layers.BatchNormalization()(low4)
    low4 = layers.Activation("relu")(low4)

    high4, low4 = OctaveConv2D(512)([high4, low4])

    high4 = layers.BatchNormalization()(high4)
    high4 = layers.Activation("relu")(high4)
    low4 = layers.BatchNormalization()(low4)
    low4 = layers.Activation("relu")(low4)

    pool4high = layers.MaxPooling2D(2)(high4)
    pool4low = layers.MaxPooling2D(2)(low4)

    high5, low5 = OctaveConv2D(1024)([pool4high, pool4low])

    high5 = layers.BatchNormalization()(high5)
    high5 = layers.Activation("relu")(high5)
    low5 = layers.BatchNormalization()(low5)
    low5 = layers.Activation("relu")(low5)

    high5 = Dropout(0.4)(high5)
    low5 = Dropout(0.4)(low5)

    high5, low5 = OctaveConv2D(1024)([high5, low5])
    high5 = layers.BatchNormalization()(high5)
    high5 = layers.Activation("relu")(high5)
    low5 = layers.BatchNormalization()(low5)
    low5 = layers.Activation("relu")(low5)

    high5 = Dropout(0.4)(high5)
    low5 = Dropout(0.4)(low5)
    
    uphigh6, uplow6 = OctaveConv2D(512, use_transpose=True, strides=(2,2))([high5,low5])

    uphigh6 = layers.BatchNormalization()(uphigh6)
    uphigh6 = layers.Activation("relu")(uphigh6)
    uplow6 = layers.BatchNormalization()(uplow6)
    uplow6 = layers.Activation("relu")(uplow6)

    merge6high = concatenate([high4,uphigh6], axis = 3)
    merge6low = concatenate([low4,uplow6], axis = 3)

    high6, low6 = OctaveConv2D(512)([merge6high,merge6low])

    high6 = layers.BatchNormalization()(high6)
    high6 = layers.Activation("relu")(high6)
    low6 = layers.BatchNormalization()(low6)
    low6 = layers.Activation("relu")(low6)

    high6, low6 = OctaveConv2D(512)([high6, low6])

    high6 = layers.BatchNormalization()(high6)
    high6 = layers.Activation("relu")(high6)
    low6 = layers.BatchNormalization()(low6)
    low6 = layers.Activation("relu")(low6)


    uphigh7, uplow7 = OctaveConv2D(256, use_transpose=True, strides=(2,2))([high6, low6])

    uphigh7 = layers.BatchNormalization()(uphigh7)
    uphigh7 = layers.Activation("relu")(uphigh7)
    uplow7 = layers.BatchNormalization()(uplow7)
    uplow7 = layers.Activation("relu")(uplow7)

    merge7high = concatenate([high3,uphigh7], axis = 3)
    merge7low = concatenate([low3,uplow7], axis = 3)

    high7, low7 = OctaveConv2D(256)([merge7high, merge7low])

    high7 = layers.BatchNormalization()(high7)
    high7 = layers.Activation("relu")(high7)
    low7 = layers.BatchNormalization()(low7)
    low7 = layers.Activation("relu")(low7)

    high7, low7 = OctaveConv2D(256)([high7, low7])

    high7 = layers.BatchNormalization()(high7)
    high7 = layers.Activation("relu")(high7)
    low7 = layers.BatchNormalization()(low7)
    low7 = layers.Activation("relu")(low7)

    uphigh8, uplow8 = OctaveConv2D(128, use_transpose=True, strides=(2,2))([high7, low7])

    uphigh8 = layers.BatchNormalization()(uphigh8)
    uphigh8 = layers.Activation("relu")(uphigh8)
    uplow8 = layers.BatchNormalization()(uplow8)
    uplow8 = layers.Activation("relu")(uplow8)

    merge8high = concatenate([high2,uphigh8], axis = 3)
    merge8low = concatenate([low2,uplow8], axis = 3)

    high8, low8 = OctaveConv2D(128)([merge8high, merge8low])

    high8 = layers.BatchNormalization()(high8)
    high8 = layers.Activation("relu")(high8)
    low8 = layers.BatchNormalization()(low8)
    low8 = layers.Activation("relu")(low8)

    high8, low8 = OctaveConv2D(128)([high8, low8])

    high8 = layers.BatchNormalization()(high8)
    high8 = layers.Activation("relu")(high8)
    low8 = layers.BatchNormalization()(low8)
    low8 = layers.Activation("relu")(low8)

    uphigh9, uplow9 = OctaveConv2D(64, use_transpose=True, strides=(2,2))([high8, low8])

    uphigh9 = layers.BatchNormalization()(uphigh9)
    uphigh9 = layers.Activation("relu")(uphigh9)
    uplow9 = layers.BatchNormalization()(uplow9)
    uplow9 = layers.Activation("relu")(uplow9)

    merge9high = concatenate([high1,uphigh9], axis = 3)
    merge9low = concatenate([low1,uplow9], axis = 3)

    high9, low9 = OctaveConv2D(64)([merge9high, merge9low])

    high9 = layers.BatchNormalization()(high9)
    high9 = layers.Activation("relu")(high9)
    low9 = layers.BatchNormalization()(low9)
    low9 = layers.Activation("relu")(low9)

    high9, low9 = OctaveConv2D(64)([high9, low9])

    high9 = layers.BatchNormalization()(high9)
    high9 = layers.Activation("relu")(high9)
    low9 = layers.BatchNormalization()(low9)
    low9 = layers.Activation("relu")(low9)

    conv9 = OctaveConv2D(32, ratio_out=0.0)([high9, low9])
    
    conv9 = layers.Activation("sigmoid")(conv9)
    conv10 = layers.Conv2D(1, 1, activation = 'sigmoid')(conv9)

    model = Model(inputs=[inputs], outputs=[conv10])
    return model
