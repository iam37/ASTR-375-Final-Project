import numpy as np
import tensorflow as tf
import pandas as pd
from tqdm import tqdm
import glob
import logging
from tensorflow.keras.utils import to_categorical
from pandas.errors import EmptyDataError

# This code is adapted from DRAGON, a CNN Jeremy Ng and I created to classify dual AGN and galaxy merger candidates in large sky survey fields. You can find the github here: https://github.com/iam37/DRAGON_CNN 
def normalize(t, f):
        # center t on its median, and divide by its full span
        t0 = np.median(t)
        span = np.ptp(t)
        span = span if span>0 else 1.0
        t = (t - t0) / span
        # shift flux so that baseline = 0
        f = f - 1.0
        return t, f

def load_lc(filepath, label, num_classes = 6):
    label_mapping = {"terrestrial_no_moon": 0, "neptunian_no_moon": 1, "jovian_no_moon": 2, "terrestrial_moon": 3, "neptunian_moon": 4, "jovian_moon": 5}
    num_label = label_mapping[label]
    lightcurves = []
    labels = []
    lc_filepaths = []
    #logging.info(f"Loading images from {filepath} with label {label}...")
    for lc_file in tqdm(glob.glob(filepath + "*.csv"), desc=f"Loading images from {filepath} with label {label}..."):
        lc_filepaths.append(lc_file)
        try:
            df = pd.read_csv(lc_file)
            times = df['time'].to_numpy()
            fluxes = df['lc_data'].to_numpy()
            t, f = normalize(times, fluxes)

            # stack time & flux into shape (1, T, 2)
            lc = np.stack([t, f], axis=1)[None,:,:]
            lightcurves.append(lc)
        except (OSError, EmptyDataError) as e:
            print(f"{e} has occured with lightcurve: {lc_file}, skipping file...")
            continue
    #lightcurves = np.asarray(images)
    y_int = [num_label] * len(lightcurves)
    labels = to_categorical(y_int, num_classes=num_classes)
    print(f"Loaded {len(lightcurves)} images with {len(labels)} labels from {filepath}")
    return lightcurves, labels, lc_filepaths
def create_lc_datasets(terr_no_moon_filepath=None, nep_no_moon_filepath = None, jovian_no_moon_filepath = None, terr_filepath = None, nep_filepath = None, jovian_filepath = None, train = 0.65, val = 0.2, test = 0.15): #Remember to change original code to account for this change!
    all_images = []
    all_labels = []
    all_filepaths = []
    if terr_no_moon_filepath:
        terr_nm_lc, terr_nm_labels, terr_nm_filepaths = load_lc(terr_no_moon_filepath, "terrestrial_no_moon")
        all_images.append(terr_nm_lc)
        all_labels.append(terr_nm_labels)
        all_filepaths.append(terr_nm_filepaths)
    if nep_no_moon_filepath:
        nep_nm_lc, nep_nm_labels, nep_nm_filepaths = load_lc(nep_no_moon_filepath, "neptunian_no_moon")
        all_images.append(nep_nm_lc)
        all_labels.append(nep_nm_labels)
        all_filepaths.append(nep_nm_filepaths)
    if jovian_no_moon_filepath:
        jov_nm_lc, jov_nm_labels, jov_nm_filepaths = load_lc(jovian_no_moon_filepath, "jovian_no_moon")
        all_images.append(jov_nm_lc)
        all_labels.append(jov_nm_labels)
        all_filepaths.append(jov_nm_filepaths)

    if terr_filepath:
        terr_lc, terr_labels, terr_filepaths = load_lc(terr_filepath, "terrestrial_moon")
        all_images.append(terr_lc)
        all_labels.append(terr_labels)
        all_filepaths.append(terr_filepaths)
    if nep_filepath:
        nep_lc, nep_labels, nep_filepaths = load_lc(nep_filepath, "neptunian_moon")
        all_images.append(nep_lc)
        all_labels.append(nep_labels)
        all_filepaths.append(nep_filepaths)
    if jovian_filepath:
        jov_lc, jov_labels, jov_filepaths = load_lc(jovian_filepath, "jovian_moon")
        all_images.append(jov_lc)
        all_labels.append(jov_labels)
        all_filepaths.append(jov_filepaths)
    
    flat_lcs  = [img for group in tqdm(all_images, desc='flattening lc') for img in group]  # each img is (1,T_i,2)
    flat_labels  = np.vstack([label for group in tqdm(all_labels, desc='flattening labels') for label in group])  # (N, C)
    flat_paths = [pth for group in tqdm(all_filepaths) for pth in group]  # list of length N

    lc_sequences = [lc.squeeze(0) for lc in tqdm(flat_lcs)]  # each is (T_i,2)
    #ragged_lightcurves  = tf.ragged.constant(lc_sequences, ragged_rank=1)

    # Dataset shuffling
    N_total = len(flat_paths)  
    idx = tf.random.shuffle(tf.range(N_total))

    row_splits = [0]
    chunks = []

    for seq in tqdm(lc_sequences, desc="Flattening sequences"):
        chunks.append(seq)  
        row_splits.append(row_splits[-1] + seq.shape[0])

    flat_values = np.vstack(chunks).astype(np.float32)  
    row_splits = np.array(row_splits, dtype=np.int64)
    ragged_lightcurves = tf.RaggedTensor.from_row_splits(flat_values, row_splits)

    #ragged_lightcurves = tf.gather(ragged_lightcurves, idx)
    labels_tf = tf.constant(flat_labels)
    paths_tf = tf.constant(flat_paths)
    idx = tf.random.shuffle(tf.range(N_total))
    shuffled_labels = tf.gather(labels_tf, idx)
    shuffled_paths = tf.gather(paths_tf, idx)   # now also a Tensor

    n_train = int(train * N_total)
    n_val   = int(val   * N_total)

    X_train = ragged_lightcurves[:n_train]
    y_train = shuffled_labels[:n_train]
    P_train = shuffled_paths[:n_train]

    X_val   = ragged_lightcurves[n_train:n_train+n_val]
    y_val   = shuffled_labels[n_train:n_train+n_val]
    P_val   = shuffled_paths[n_train:n_train+n_val]

    X_test  = ragged_lightcurves[n_train+n_val:]
    y_test  = shuffled_labels[n_train+n_val:]
    P_test  = shuffled_paths[n_train+n_val:]

    

    return (X_train, y_train, P_train), (X_val, y_val, P_val), (X_test, y_test, P_test)
