import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from keras.regularizers import l2
from time import time

class LSTM_model():
    def __init__(self, M, T, N, num_classes = 1, lam = 3e-2, dropout = 0.0, recurrent_dropout = 0.0, learning_rate = 1e-5, display_architecture = True):
        self.M = M
        self.T = T
        self.N = N
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.display_architecture = display_architecture
        self.lam = lam
        self.dropout = dropout
        self.recurrent_dropout = recurrent_dropout
    def create_neural_network(self):
        model = Sequential()
        model.add(LSTM(input_shape=(T, N), units=8,
               activation='tanh', recurrent_activation='hard_sigmoid',
               kernel_regularizer=l2(self.lam), recurrent_regularizer=l2(self.lam),
               dropout=dropout, recurrent_dropout=recurrent_dropout,
               return_sequences=True, return_state=False,
               stateful=False, unroll=False
              ))
        model.add(BatchNormalization())
        model.add(LSTM(units=16, activation='tanh', recurrent_activation='hard_sigmoid',
               kernel_regularizer=l2(self.lam), recurrent_regularizer=l2(self.lam),
               dropout=dropout, recurrent_dropout=recurrent_dropout,
               return_sequences=True, return_state=False,
               stateful=False, unroll=False))
        model.add(BatchNormalization())
        model.add(LSTM(units=16, activation='tanh', recurrent_activation='hard_sigmoid',
               kernel_regularizer=l2(self.lam), recurrent_regularizer=l2(self.lam),
               dropout=dropout, recurrent_dropout=recurrent_dropout,
               return_sequences=True, return_state=False,
               stateful=False, unroll=False))
        model.add(BatchNormalization())
        model.add(LSTM(units=16, activation='tanh', recurrent_activation='hard_sigmoid',
               kernel_regularizer=l2(self.lam), recurrent_regularizer=l2(self.lam),
               dropout=dropout, recurrent_dropout=recurrent_dropout,
               return_sequences=True, return_state=False,
               stateful=False, unroll=False))
        model.add(BatchNormalization())
        model.add(LSTM(units=16, activation='tanh', recurrent_activation='hard_sigmoid',
               kernel_regularizer=l2(self.lam), recurrent_regularizer=l2(self.lam),
               dropout=dropout, recurrent_dropout=recurrent_dropout,
               return_sequences=True, return_state=False,
               stateful=False, unroll=False))
        model.add(BatchNormalization())
        model.add(Dense(16, activation = tf.nn.leaky_relu()))
        model.add(Dense(8, activation = tf.nn.leaky_relu()))
        model.add(Dense(1, activation = 'sigmoid'))
        lr_scheduler = tf.keras.optimizers.schedules.CosineDecay(self.learning_rate, 1000, alpha = 1e-3)
        optimized = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

        model.compile(optimizer=optimized, loss=tf.keras.losses.BinaryCrossentropy(),
                      metrics=[tf.keras.metrics.BinaryAccuracy(name='binary_accuracy'),
                               tf.keras.metrics.Recall(name='recall'),
                               tf.keras.metrics.Precision(name='precision'),
                               tf.keras.metrics.F1Score], run_eagerly=True)
        if display_architecture:
            model.summary()
        return model
        