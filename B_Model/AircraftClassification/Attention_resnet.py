import tensorflow as tf
from keras.layers import *

from B_Model.AbstractModel import Model as AbstactModel
from B_Model.Utils.TF_Modules import *

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

    name = ("Attention_Resnet")

    def __init__(self, CTX: dict):
        """
        Generate model architecture
        Define loss function
        Define optimizer
        """

        self.CTX = CTX

        # prepare input shapes
        x_input_shape = (self.CTX["INPUT_LEN"], self.CTX["FEATURES_IN"])
        if (CTX["ADD_TAKE_OFF_CONTEXT"]): takeoff_input_shape = (self.CTX["INPUT_LEN"], self.CTX["FEATURES_IN"])
        if (CTX["ADD_MAP_CONTEXT"]): map_input_shape = (self.CTX["IMG_SIZE"], self.CTX["IMG_SIZE"], 3)

        # generate layers
        x = tf.keras.Input(shape=x_input_shape, name='input')
        inputs = [x]
        outputs = []

        adsb_module_inputs = [x]
        if (CTX["ADD_TAKE_OFF_CONTEXT"]):
            takeoff = tf.keras.Input(shape=takeoff_input_shape, name='takeoff')
            inputs.append(takeoff)

            self.takeoff_module = TakeOffModule(self.CTX)
            takeoff_ctx = self.takeoff_module(takeoff)
            adsb_module_inputs.append(takeoff_ctx)
            self.TAKEOFF = len(outputs)

        if (CTX["ADD_MAP_CONTEXT"]):
            map = tf.keras.Input(shape=map_input_shape, name='map')
            inputs.append(map)

            self.map_module = MapModule(self.CTX)
            map_ctx = self.map_module(map)
            adsb_module_inputs.append(map_ctx)
            self.MAP = len(outputs)

        self.ads_b_module = ADS_B_Module(self.CTX)
        proba = self.ads_b_module(adsb_module_inputs)
        outputs.insert(0, proba)

        # generate model
        self.model = tf.keras.Model(inputs, outputs)

        # define loss and optimizer
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


def AttentionMoudule(inputs, head_size, num_heads, ff_dim, dropout=0):
    # Normalization and Attention
    x = LayerNormalization(epsilon=1e-6)(inputs)
    x = MultiHeadAttention(
        key_dim=head_size, num_heads=num_heads, dropout=dropout
    )(x, x)
    x = Dropout(dropout)(x)
    res = x + inputs

    # Feed Forward Part
    x = LayerNormalization(epsilon=1e-6)(res)
    x = Dense(inputs.shape[-1])(x)
    x = Dropout(dropout)(x)
    return x + res

def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
    # Normalization and Attention
    x = LayerNormalization(epsilon=1e-6)(inputs)
    x = MultiHeadAttention(
        key_dim=head_size, num_heads=num_heads, dropout=dropout
    )(x, x)
    x = Dropout(dropout)(x)
    res = x + inputs

    # Feed Forward Part
    x = LayerNormalization(epsilon=1e-6)(res)
    x = Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(x)
    x = Dropout(dropout)(x)
    x = Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
    return x + res

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
        if self.CTX['TAKE_OFF_ATTENTION']:
            x = transformer_encoder(x, head_size=self.CTX['KEY_DIM'], num_heads=self.CTX['NUM_HEADS'],
                                 ff_dim=self.CTX['FF_DIM'], dropout=0.2)
        for layer in self.convNN:
            x = layer(x)
        return x


class ResidualBlock(tf.Module):
    def __init__(self, filters, kernel_size, padding, strides=(1, 1)):
        super(ResidualBlock, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides, activation='relu')
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.conv2 = tf.keras.layers.Conv2D(3, kernel_size, padding=padding, activation=None)
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.add = tf.keras.layers.Add()
        self.relu = tf.keras.layers.Activation('relu')

    def __call__(self, x):
        identity = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.add([x, identity])
        x = self.relu(x)
        return x


class MapModule(tf.Module):

    def __init__(self, CTX):
        self.CTX = CTX
        self.dropout = self.CTX["DROPOUT"]
        self.outs = self.CTX["FEATURES_OUT"]

        self.convNN = tf.keras.Sequential([
            ResidualBlock(16, 3, padding=self.CTX["MODEL_PADDING"]),
            tf.keras.layers.MaxPooling2D(),
            ResidualBlock(64, 3, padding=self.CTX["MODEL_PADDING"]),
            tf.keras.layers.Conv2D(32, (2, 2), (2, 2)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dropout(self.dropout),
            tf.keras.layers.Dense(self.outs, activation=None)  # Assume self.outs is the final layer size
        ])

    def __call__(self, x):
        return self.convNN(x)


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

        self.multi_head_attention = MultiHeadAttention(num_heads=8, key_dim=32)
        self.layer_norm = LayerNormalization(epsilon=1e-6)

        self.cat = Concatenate()
        self.catmap = Concatenate()

        convNN = []
        convNN.append(DenseModule(128, dropout=self.dropout))
        convNN.append(Dense(self.outs, activation="linear", name="prediction"))

        self.preNN = preNN
        self.postMap = postMap
        self.convNN = convNN
        self.probability = Activation(CTX["ACTIVATION"], name=CTX["ACTIVATION"])
        self.sequence_length = 24
        self.embedding_dim = 32

    def attention(self, feature):
        pass

    def __call__(self, x):

        adsb = x.pop(0)
        if (self.CTX["ADD_TAKE_OFF_CONTEXT"]):
            takeoff = x.pop(0)
        if (self.CTX["ADD_MAP_CONTEXT"]):
            map = x.pop(0)

        # preprocess
        if self.CTX["ADSB_ATTENTION"]:
            x = transformer_encoder(adsb, head_size=self.CTX['KEY_DIM'], num_heads=self.CTX['NUM_HEADS'],
                             ff_dim=self.CTX['FF_DIM'], dropout=0.2)

        for layer in self.preNN:
            x = layer(x)
        # ...
        for layer in self.postMap:
            x = layer(x)

        # concat takeoff and map
        cat = [x]

        if self.CTX["ADD_MAP_CONTEXT"]:
            cat.append(map)

        if self.CTX["ADD_TAKE_OFF_CONTEXT"]:
            cat.append(takeoff)

        if self.CTX["MERGE_ATTENTION"]:
            combined = tf.stack([x, map, takeoff], axis=-1)  # 结果形状为(256, 3)
            x = transformer_encoder(combined, head_size=self.CTX['KEY_DIM'], num_heads=self.CTX['NUM_HEADS'],
                                 ff_dim=self.CTX['FF_DIM'], dropout=0.2)
            x = Flatten()(x)
        else:
            x = self.cat([x, map, takeoff])

        # get prediction
        for layer in self.convNN:
            x = layer(x)
        x = self.probability(x)
        return x

