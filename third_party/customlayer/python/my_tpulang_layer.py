import numpy as np
import transform.TpuLang as tpul

class absAdd:
    @staticmethod
    def native(data, b):
        return np.abs(data) + b
    @staticmethod
    def tpulang(inputs, b, dtype="float32"):
        def shape_func(tensors_in:list):
            return [tensors_in[0].shape]
        params = {"b": b}
        outs = tpul.custom(
            tensors_in=inputs,
            shape_func=shape_func,
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
        def shape_func(tensors_in:list):
            return [tensors_in[0].shape]
        params = {"b": b}
        outs = tpul.custom(
            tensors_in=inputs,
            shape_func=shape_func,
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
        def shape_func(tensors_in:list):
            return [tensors_in[0].shape]
        params = {"order": [2, 1, 0]}
        outs = tpul.custom(
            tensors_in=inputs,
            shape_func=shape_func,
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
        def shape_func(tensors_in):
            # the shape inference function
            # return the list of output shapes
            return [[tensors_in[0].shape[0], tensors_in[0].shape[1], hnew, wnew]]
        params = {"hoffset": hoffset, "woffset": woffset, "hnew": hnew, "wnew": wnew}
        outs = tpul.custom(
            tensors_in=inputs,
            shape_func=shape_func,
            # op_name should be consistent with the backend
            op_name="crop",
            params=params,
            out_dtypes=[dtype])
        return outs
