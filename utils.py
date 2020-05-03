from mne.time_frequency import psd_welch
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import itertools

# https://mne.tools/dev/auto_tutorials/sample-datasets/plot_sleep.html#sphx-glr-auto-tutorials-sample-datasets-plot-sleep-py%00
def eeg_power_band(epochs):
    """EEG relative power band feature extraction.

    This function takes an ``mne.Epochs`` object and creates EEG features based
    on relative power in specific frequency bands that are compatible with
    scikit-learn.

    Parameters
    ----------
    epochs : Epochs
        The data.

    Returns
    -------
    X : numpy array of shape [n_samples, 5]
        Transformed data.
    """
    # specific frequency bands
    FREQ_BANDS = {"delta": [0.5, 4.5],
                  "theta": [4.5, 8.5],
                  "alpha": [8.5, 11.5],
                  "sigma": [11.5, 15.5],
                  "beta": [15.5, 30]}

    psds, freqs = psd_welch(epochs, picks='eeg', fmin=0.5, fmax=30.)
    # Normalize the PSDs
    psds /= np.sum(psds, axis=-1, keepdims=True)

    X = []
    for fmin, fmax in FREQ_BANDS.values():
        psds_band = psds[:, :, (freqs >= fmin) & (freqs < fmax)].mean(axis=-1)
        X.append(psds_band.reshape(len(psds), -1))

    return np.concatenate(X, axis=1)


def get_series_and_features(epochs, channels):
    """
    this is to extract label, raw signal and calcuate time domain feature from decrypted edf data
    """
    epoch_dfs = []
    time_features = {"time_min": [], "time_max": [], "time_std": [], "time_kurtosis": [], "time_median": []}
    labels = np.ones(len(epochs))
    for i in range(len(epochs)):
        print("{}th epoch started to be processed.".format(i))
        epoch = epochs[i]
        epoch_df = epoch.to_data_frame()
        epoch_df = epoch_df[channels]
        epoch_dfs.append(epoch_df)
        time_features["time_min"].append(epoch_df.min())
        time_features["time_max"].append(epoch_df.max())
        time_features["time_std"].append(epoch_df.std())
        time_features["time_kurtosis"].append(epoch_df.kurtosis())
        time_features["time_median"].append(epoch_df.median())
        label = list(epoch.event_id.values())[0]
        labels[i] = label
    for key, value in time_features.items():
        new_df = pd.concat(time_features[key], axis=1).transpose()
        new_df.reset_index(drop=True, inplace=True)
        orig_columns = list(new_df.columns)
        new_columns = [column + "_" + key for column in orig_columns]
        new_df.columns = new_columns
        time_features[key] = new_df
    dfs = list(time_features.values())
    time_feature_df = pd.concat(dfs, axis=1)
    return epoch_dfs, time_feature_df, labels


def to_tensor(list_data, device, type="float"):
    """
    function to change from a list to corresponding tenor
    """
    if type == "float":
        return torch.from_numpy(list_data).float().to(device, non_blocking=True)
    elif type == "int":
        return torch.from_numpy(list_data).int().to(device, non_blocking=True)
    elif type == "long":
        return torch.from_numpy(list_data).long().to(device, non_blocking=True)


def plot_confusion_matrix(true_y_label, pred_y_label, class_names, tag, dataset):
    """
    function to plot confusion matrix
    """
    internalMatrixFunction = confusion_matrix(true_y_label, pred_y_label)
    filename = "plot/confusion_" + tag + "_" + str(dataset) + ".png"
    cm = internalMatrixFunction.astype('float') / internalMatrixFunction.sum(axis=1)[:, np.newaxis]
    fig = plt.figure(figsize=(12, 8), tight_layout=True)
    ax = fig.add_subplot(111)
    cax = plt.imshow(cm, interpolation='nearest', cmap=plt.get_cmap('Blues'))
    format_type = '.2f'
    max_limit = cm.max() / 2.
    for x, y in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(y, x, format(cm[x, y], format_type), horizontalalignment="center",
                 color="white" if cm[x, y] > max_limit else "black")
    plt.title('Confusion matrix of the classifier')
    fig.colorbar(cax)
    checkered_boxes = np.arange(len(class_names))
    plt.xticks(checkered_boxes, class_names)
    plt.yticks(checkered_boxes, class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    fig.savefig(filename, dpi=fig.dpi)
    pass


