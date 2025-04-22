import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization, InputLayer, Conv1D, Masking, MaxPooling1D
from tensorflow.keras.optimizers import Adam
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
#import tensorflow_addons as tfa
from keras.regularizers import l2
from time import time

class LSTM_model():
    def __init__(self, N=2, num_classes = 6, lam = 3e-2, dropout = 0.1, recurrent_dropout = 0.1, learning_rate = 1e-5, display_architecture = True):
        self.N = N
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.display_architecture = display_architecture
        self.lam = lam
        self.dropout = dropout
        self.recurrent_dropout = recurrent_dropout
    def create_neural_network(self):
        model = Sequential()
        #model.add(Masking(mask_value=2.0, input_shape=(None, self.N)))
        #model.add(LayerNormalization())
        model.add(InputLayer(input_shape=(None, self.N)))
        model.add(Conv1D(filters=32, kernel_size=5, strides=1, padding="same", activation=tf.nn.leaky_relu))
        model.add(MaxPooling1D(pool_size=2))
        model.add(LSTM(units=16, activation='tanh', recurrent_activation='sigmoid',
               kernel_regularizer=l2(self.lam), recurrent_regularizer=l2(self.lam),
               dropout=self.dropout, recurrent_dropout=self.recurrent_dropout,
               return_sequences=True, return_state=False,
               stateful=False, unroll=False))
        model.add(LSTM(units=16, activation='tanh', recurrent_activation='sigmoid',
               kernel_regularizer=l2(self.lam), recurrent_regularizer=l2(self.lam),
               dropout=self.dropout, recurrent_dropout=self.recurrent_dropout,
               return_sequences=True, return_state=False,
               stateful=False, unroll=False))
        #model.add(LayerNormalization())
        model.add(LSTM(units=16, activation='tanh', recurrent_activation='sigmoid',
               kernel_regularizer=l2(self.lam), recurrent_regularizer=l2(self.lam),
               dropout=self.dropout, recurrent_dropout=self.recurrent_dropout,
               return_sequences=False, return_state=False,
               stateful=False, unroll=False))
        model.add(Dense(16, activation = tf.nn.leaky_relu))
        model.add(Dense(8, activation = tf.nn.leaky_relu))
        model.add(Dense(self.num_classes, activation = 'softmax'))
        lr_scheduler = tf.keras.optimizers.schedules.CosineDecay(self.learning_rate, 1000, alpha = 1e-3)
        optimized = tf.keras.optimizers.Adam(learning_rate=lr_scheduler)

        model.compile(optimizer=optimized, loss=tf.keras.losses.CategoricalCrossentropy(),
                      metrics=[tf.keras.metrics.CategoricalAccuracy(name='accuracy'),
                               tf.keras.metrics.Recall(name='recall'),
                               tf.keras.metrics.Precision(name='precision')], run_eagerly=False)
        if self.display_architecture:
            model.summary()
        return model
        