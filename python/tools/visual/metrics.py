#!/usr/bin/env python3
import numpy as np
from collections import OrderedDict

metrics = OrderedDict()


def register(func):
    name = func.__name__
    if name in metrics:
        raise Exception("Duplicate name: '{}'.".format(name))
    metrics[name] = func
    return func


@register
def cos(a, b):
    a_vec = a.flatten()
    b_vec = b.flatten()
    denom = float(np.linalg.norm(a_vec) * np.linalg.norm(b_vec))
    a_dot_b = float(np.dot(a_vec, b_vec))
    cos_dis = a_dot_b / (denom + np.finfo(np.float32).eps)
    if np.isinf(cos_dis):
        return -1
    return cos_dis


@register
def mape(ref_data, test_data):
    if ref_data.size != test_data.size:
        Warning('The data shape does not match!')
    ref_d_vec = ref_data.flatten()
    test_d_vec = test_data.flatten()
    diff = ref_d_vec - test_d_vec
    # If the reference data is less than 1, the relative error will be enlarged.
    # Here using absolute error when the reference data is less than 1.
    use_abs_ind = np.abs(ref_d_vec) < 1
    mape_abs = np.abs(diff[use_abs_ind])
    mape_rela = np.abs(diff[~use_abs_ind] / ref_d_vec[~use_abs_ind])
    mape_sum = np.sum(mape_abs) + np.sum(mape_rela)
    return mape_sum / (mape_abs.size + mape_rela.size)

@register
def dist(a, b):
    return 1.0 / (1.0 + np.sqrt(np.sum(np.square(a - b))))

@register
def sqnr(ref_data, test_data, remove_zero = False):
    import math
    raw = ref_data.flatten()
    dequant = test_data.flatten()

    if remove_zero is True:
        idx = dequant != 0
        raw = raw[idx]
        dequant = dequant[idx]

    noise = raw - dequant
    avg_raw = np.sum(raw) / raw.size
    avg_noise = np.sum(noise) / noise.size

    raw_zero_mean = raw - avg_raw
    noise_zero_mean = noise - avg_noise

    var_raw_zero_mean = np.sum(np.square(raw_zero_mean))
    var_noise_zero_mean = np.sum(np.square(noise_zero_mean))

    if var_noise_zero_mean == 0.0:
        return math.inf

    sqnr = 10 * np.log10(var_raw_zero_mean / var_noise_zero_mean)

    return sqnr

def kurtosis(data):
    import numpy as np

    mean=np.mean(data)
    n=data.size
    if n <= 3:
        return 3
    kuru=np.sum(np.power((data-mean),4))/n
    kurd=np.power(np.sum(np.power((data-mean),2))/n,2)
    kur=kuru/kurd
    return kur

def skewness(data):
    import numpy as np

    mean=np.mean(data)
    std=np.std(data)
    n=data.size
    ske=np.sum(np.power((data-mean)/std,3))/n/np.power(std,3)
    return ske
