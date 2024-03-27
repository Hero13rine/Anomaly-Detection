import tensorflow as tf
from keras.layers import *

from B_Model.AbstractModel import Model as AbstactModel
from B_Model.Utils.TF_Modules import *
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as keras

import numpy as np

import os
from random import randint


class Model(AbstactModel):
    """
    Convolutional neural network model for
    aircraft classification based on
    recordings of ADS-B data fragment.

    Parameters:
    ------------

    CTX: dict
        The hyperparameters context


    Attributes:
    ------------

    name: str (MENDATORY)
        The name of the model for mlflow logs

    Methods:
    ---------

    predict(x): (MENDATORY)
        return the prediction of the model

    compute_loss(x, y): (MENDATORY)
        return the loss and the prediction associated to x, y and y_

    training_step(x, y): (MENDATORY)
        do one training step.
        return the loss and the prediction of the model for this batch

    visualize(save_path):
        Generate a visualization of the model's architecture
    """

    name = "Unet"

    def __init__(self, CTX: dict):
        """
        Generate model architecture
        Define loss function
        Define optimizer
        """

        self.CTX = CTX

        # prepare input shapes

        x_input_shape = (128, 128, 3)

        inputs = []
        outputs = []

        mapinput = tf.keras.Input(shape=x_input_shape, name='map')

        self.map_module = MapModule(self.CTX)
        map_ctx = self.map_module(mapinput)
        adsb_module_inputs = map_ctx
        self.MAP = len(outputs)

        self.ads_b_module = MapModule(self.CTX)
        proba = self.ads_b_module(adsb_module_inputs)
        outputs.insert(0, proba)

        # generate model
        self.model = tf.keras.Model(mapinput, outputs)

        # define loss and outputs
        self.loss = tf.keras.losses.MeanSquaredError()
        self.opt = tf.keras.optimizers.Adam(learning_rate=self.CTX["LEARNING_RATE"])

        self.nb_train = 0

    def predict(self, x):
        """
        Make prediction for x
        """
        return self.model(x)

    def compute_loss(self, x, y):
        """
        Make a prediction and compute the loss
        that will be used for training
        """
        y_ = self.model(x)
        loss = self.loss(y_, y)
        return loss, y_

    def training_step(self, x, y):
        """
        Do one forward pass and gradient descent
        for the given batch
        """

        skip_w = self.CTX["SKIP_CONNECTION"]

        with tf.GradientTape(watch_accessed_variables=True) as tape:
            y_ = self.model(x)
            loss = self.loss(y_, y)

            gradients = tape.gradient(loss, self.model.trainable_variables)
            self.opt.apply_gradients(zip(gradients, self.model.trainable_variables))

        self.nb_train += 1
        return loss, y_

    def visualize(self, save_path="./_Artifacts/"):
        """
        Generate a visualization of the model's architecture
        """

        filename = os.path.join(save_path, self.name + ".png")
        tf.keras.utils.plot_model(self.model, to_file=filename, show_shapes=True)

    def getVariables(self):
        """
        Return the variables of the model
        """
        return self.model.trainable_variables

    def setVariables(self, variables):
        """
        Set the variables of the model
        """
        for i in range(len(variables)):
            self.model.trainable_variables[i].assign(variables[i])


class MapModule(tf.Module):

    def __init__(self, CTX):
        self.CTX = CTX
        self.layers = 1
        self.dropout = self.CTX["DROPOUT"]
        self.outs = self.CTX["FEATURES_OUT"]

    def __call__(self, x):
        # 下采样1
        conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(x)
        conv1 = BatchNormalization()(conv1)
        conv1 = Dropout(0.1)(conv1)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

        conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
        conv2 = BatchNormalization()(conv2)
        conv2 = Dropout(0.1)(conv2)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

        up1 = UpSampling2D(size=(2, 2))(pool2)
        merge1 = concatenate([conv2, up1], axis=3)
        conv3 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge1)
        conv3 = BatchNormalization()(conv3)

        up2 = UpSampling2D(size=(2, 2))(conv3)
        merge2 = concatenate([conv1, up2], axis=3)
        conv4 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge2)
        conv4 = BatchNormalization()(conv4)
        conv5 = Conv2D(2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
        dp = Dropout(0.1)(conv5)
        pool1 = MaxPooling2D(pool_size=(2, 2))(dp)
        fl = Flatten(name='fla1')(pool1)
        dense_output1 = Dense(256, activation='relu')(fl)
        """dropout1 = Dropout(0.5)(dense_output1)
        dense_output2 = Dense(128, activation='relu')(dropout1)
        dropout2 = Dropout(0.5)(dense_output2)
        
        final_output = Dense(3, activation='softmax')(dropout2)"""

        return dense_output1


