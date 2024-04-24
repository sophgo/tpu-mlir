import numpy as np
import transform.TpuLang as tpul

def get_dtype2int(dtype):
    if dtype == 'float32':
        return 1
    if dtype == 'float16':
        return 2
    if dtype == 'int8':
        return 3
    if dtype == 'uint8':
       return 4
    assert 0,"not support now!"
class absAdd:
    @staticmethod
    def native(data, b):
        return np.abs(data) + b
    @staticmethod
    def tpulang(inputs, b, dtype="float32"):
        params = {"b": b}
        outs = tpul.custom(
            tensors_in=inputs,
            # op_name should be consistent with the backend
            op_name="absadd",
            params=params,
            out_dtypes=[dtype])
        return outs

class ceilAdd:
    @staticmethod
    def native(data, b):
        return np.ceil(data) + b
    @staticmethod
    def tpulang(inputs, b, dtype="float32"):
        params = {"b": b}
        outs = tpul.custom(
            tensors_in=inputs,
            # op_name should be consistent with the backend
            op_name="ceiladd",
            params=params,
            out_dtypes=[dtype])
        return outs

class swapChannel:
    @staticmethod
    def native(data):
        return data[:, [2, 1, 0], :, :]
    @staticmethod
    def tpulang(inputs, dtype="float32"):
        params = {"order": [2, 1, 0]}
        outs = tpul.custom(
            tensors_in=inputs,
            # op_name should be consistent with the backend
            op_name="swapchannel",
            params=params,
            out_dtypes=[dtype])
        return outs

class crop:
    @staticmethod
    def native(data, hoffset, woffset, hnew, wnew):
        data_crop = np.zeros([data.shape[0], data.shape[1], hnew, wnew])
        for n in range(data.shape[0]):
            for c in range(data.shape[1]):
                for i in range(hnew):
                    iold = i + hoffset
                    for j in range(wnew):
                        jold = j + woffset
                        data_crop[n, c, i, j] = data[n, c, iold, jold]
        return data_crop
    @staticmethod
    def tpulang(inputs, hoffset, woffset, hnew, wnew, dtype="float32"):
        params = {"hoffset": hoffset, "woffset": woffset, "hnew": hnew, "wnew": wnew}
        outs = tpul.custom(
            tensors_in=inputs,
            # op_name should be consistent with the backend
            op_name="crop",
            params=params,
            out_dtypes=[dtype])
        return outs

class preprocess:
    @staticmethod
    def native(data, scale, mean):
        result = (data - mean) * scale
        print(" result: ",result)
        return result
    @staticmethod
    def tpulang(inputs, scale, mean, dtype="float32"):
        params = {"scale": scale, "mean": mean, "odtype" : get_dtype2int(dtype)}
        outs = tpul.custom(
            tensors_in=inputs,
            # op_name should be consistent with the backend
            op_name="preprocess",
            params=params,
            out_dtypes=[dtype])
        return outs

class cpuTopk:
    @staticmethod
    def native(data, axis, k):
        sorted_arr = np.sort(data, axis=axis)
        arr_descending = np.flip(sorted_arr, axis=axis)
        if axis == 1:
            return arr_descending[:, :k]
        elif axis == 0:
            return arr_descending[:k, :]

    @staticmethod
    def tpulang(inputs, axis, k, dtype="float32"):
        params = {"axis": axis, "K": k}
        outs = tpul.custom(
            tensors_in=inputs,
            # op_name should be consistent with the backend
            op_name="ap.topk",
            params=params,
            out_dtypes=[dtype])
        return outs