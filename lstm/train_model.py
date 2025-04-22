import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import click
import os

import sys
sys.path.append("datasets/")
sys.path.append("lstm/")
from load_datasets import create_lc_datasets
from create_model import LSTM_model

@click.command()
@click.option('--learning-rate',    '-lr',  default=1e-5, type=float, help='Initial learning rate.')
@click.option('--lam',              '-l',   default=1e-4, type=float, help='L2 regularization.')
@click.option('--dropout',          '-d',   default=0.1,  type=float, help='Dropout rate.')
@click.option('--rec_dropout', default=0.1, type=float, help='Recurrent dropout rate.')
@click.option('--batch-size',       '-b',   default=128,  type=int,   help='Batch size for training.')
@click.option('--epochs',           '-e',   default=10,   type=int,   help='Number of epochs to train.')

def train_model(learning_rate, lam, dropout, rec_dropout, batch_size, epochs):
    train_tuple, val_tuple, test_tuple = create_lc_datasets(terr_no_moon_filepath="datasets/terrestrial_lightcurves_no_moon/", nep_no_moon_filepath = "datasets/neptunian_lightcurves_no_moon/", jovian_no_moon_filepath = "datasets/jovian_lightcurves_no_moon/", terr_filepath = "datasets/terrestrial_lightcurves/", nep_filepath = "datasets/neptunian_lightcurves/", jovian_filepath = "datasets/jovian_lightcurves/")
    X_train, y_train, train_filepaths = train_tuple
    X_val, y_val, val_filepaths = val_tuple
    X_test, y_test, test_filepaths = test_tuple

    X_train_pad = X_train.to_tensor(default_value=1.0)
    X_val_pad   = X_val.to_tensor(default_value=1.0)


    print("GPUs:", tf.config.list_physical_devices("GPU"))
    print("Built with CUDA:", tf.test.is_built_with_cuda())
    print("GPU support:", tf.test.is_gpu_available())
    
    lstm_model = LSTM_model(dropout = dropout, lam=lam, recurrent_dropout=rec_dropout, learning_rate = learning_rate).create_neural_network()
    history = lstm_model.fit(X_train_pad, y_train, batch_size = batch_size, epochs = epochs, verbose = 1, validation_data = (X_val_pad, y_val), shuffle=True)
    save_filepath = "saved_models/"
    os.makedirs(save_filepath, exist_ok=True)
    lstm_model.save(f"{save_filepath}_saved_model_{epochs}_{batch_size}_{learning_rate}_{dropout}_{lam}.keras", overwrite=True)
    return history, model

if __name__=="__main__":
    train_model()
