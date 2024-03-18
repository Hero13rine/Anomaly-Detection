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


class TakeOffModule(tf.Module):
    CTX_SIZE = 128

    def __init__(self, CTX):
        self.CTX = CTX
        self.layers = self.CTX["LAYERS"]
        self.dropout = self.CTX["DROPOUT"]
        self.outs = self.CTX["FEATURES_OUT"]

        convNN = []
        for _ in range(self.layers):
            convNN.append(Conv1DModule(32, 3, padding=self.CTX["MODEL_PADDING"]))
        convNN.append(MaxPooling1D())
        for _ in range(self.layers):
            convNN.append(Conv1DModule(64, 3, padding=self.CTX["MODEL_PADDING"]))
        convNN.append(MaxPooling1D())
        for _ in range(self.layers):
            convNN.append(Conv1DModule(256, 3, padding=self.CTX["MODEL_PADDING"]))
        convNN.append(Flatten())
        convNN.append(DenseModule(256, dropout=self.dropout))

        self.convNN = convNN

    def __call__(self, x):
        for layer in self.convNN:
            x = layer(x)
        return x


class MapModule(tf.Module):

    def __init__(self, CTX):
        self.CTX = CTX
        self.layers = 1
        self.dropout = self.CTX["DROPOUT"]
        self.outs = self.CTX["FEATURES_OUT"]

    def __call__(self, x):
        conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(x)
        conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
        conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
        conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
        conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
        conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
        conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
        conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
        drop4 = Dropout(0.5)(conv4)
        pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

        conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
        conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
        drop5 = Dropout(0.5)(conv5)

        up6 = Conv2D(512, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
            UpSampling2D(size=(2, 2))(drop5))
        merge6 = concatenate([drop4, up6], axis=3)
        conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
        conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)

        up7 = Conv2D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
            UpSampling2D(size=(2, 2))(conv6))
        merge7 = concatenate([conv3, up7], axis=3)
        conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
        conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)

        up8 = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
            UpSampling2D(size=(2, 2))(conv7))
        merge8 = concatenate([conv2, up8], axis=3)
        conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
        conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)

        up9 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
            UpSampling2D(size=(2, 2))(conv8))
        merge9 = concatenate([conv1, up9], axis=3)
        conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge9)
        conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
        conv9 = Conv2D(2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
        conv10 = Conv2D(1, 1, activation='sigmoid')(conv9)  # 降维
        pool_out = MaxPooling2D(pool_size=(2, 2))(conv10)
        dense_in = Reshape((64, 64), name='reshape')(pool_out)
        dense_input = Flatten()(dense_in)
        dense_output = Dense(256, activation='relu')(dense_input)
        dropout = Dropout(0.5)(dense_output)
        dense_output = Dense(128, activation='relu')(dropout)
        dropout = Dropout(0.5)(dense_output)
        output = Flatten()(dropout)
        return output


class ADS_B_Module(tf.Module):

    def __init__(self, CTX):
        self.CTX = CTX
        self.layers = self.CTX["LAYERS"]
        self.dropout = self.CTX["DROPOUT"]
        self.outs = self.CTX["FEATURES_OUT"]

        preNN = []
        for _ in range(self.layers):
            preNN.append(Conv1DModule(256, 3, padding=self.CTX["MODEL_PADDING"]))
        preNN.append(MaxPooling1D())

        postMap = []
        for _ in range(self.layers):
            postMap.append(Conv1DModule(256, 3, padding=self.CTX["MODEL_PADDING"]))
        postMap.append(Flatten())
        postMap.append(DenseModule(256, dropout=self.dropout))

        self.cat = Concatenate()
        self.catmap = Concatenate()

        convNN = []
        convNN.append(DenseModule(128, dropout=self.dropout))
        convNN.append(Dense(self.outs, activation="linear", name="prediction"))

        self.preNN = preNN
        self.postMap = postMap
        self.convNN = convNN
        self.probability = Activation(CTX["ACTIVATION"], name=CTX["ACTIVATION"])

    def __call__(self, x):

        adsb = x.pop(0)
        if (self.CTX["ADD_TAKE_OFF_CONTEXT"]):
            takeoff = x.pop(0)
        if (self.CTX["ADD_MAP_CONTEXT"]):
            map = x.pop(0)

        # preprocess
        x = adsb
        for layer in self.preNN:
            x = layer(x)
        # ...
        for layer in self.postMap:
            x = layer(x)

        # concat takeoff and map
        cat = [x]
        if (self.CTX["ADD_MAP_CONTEXT"]):
            cat.append(map)
        if (self.CTX["ADD_TAKE_OFF_CONTEXT"]):
            cat.append(takeoff)

        x = self.cat([x, map, takeoff])

        # get prediction
        for layer in self.convNN:
            x = layer(x)
        x = self.probability(x)
        return x

# global accuracy mean :  92.0 ( 575 / 625 )
# global accuracy count :  92.2 ( 576 / 625 )
# global accuracy max :  87.2 ( 545 / 625 )
