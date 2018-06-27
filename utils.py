from __future__ import division
import numpy as np
import pywt


def wavelet_denoising(arr):
    cA, cD = pywt.dwt(arr, 'sym4')
    cA2, cD2 = pywt.dwt(cA, 'sym4')
    cA3, cD3 = pywt.dwt(cA2, 'sym4')
    cA4, cD4 = pywt.dwt(cA3, 'sym4')
    rev4 = pywt.idwt(cA4, None, 'sym4')
    rev3 = pywt.idwt(rev4, None, 'sym4')
    rev2 = pywt.idwt(rev3, None, 'sym4')
    rev1 = pywt.idwt(rev2, None, 'sym4')
    return rev1[:len(arr)]


def new_dataset(dataset, step_size):
    data_X, data_Y = [], []
    for i in range(len(dataset) - step_size):
        a = dataset[i:(i + step_size), 0]
        data_X.append(a)
        data_Y.append(dataset[i + step_size, 0])
    return np.array(data_X), np.array(data_Y)


def theil(arr_true, arr_pred):
    T = len(arr_true)
    val_top = 0
    val_pre = 0
    val_true = 0
    for i in range(T):
        val_top = val_top + ((arr_pred[i] - arr_true[i]) ** 2)
        val_pre = val_pre + (arr_pred[i] ** 2)
        val_true = val_true + (arr_true[i] ** 2)

    val_top = np.sqrt(val_top / T)
    val_pre = np.sqrt(val_pre / T)
    val_true = np.sqrt(val_true / T)
    return (val_top) / (val_pre + val_true)


def MAPE(arr_true, arr_pred):
    return np.mean(np.abs((arr_true - arr_pred) / arr_true)) * 100