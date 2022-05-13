from .npz_compare import npz_compare, load_op_info, dequantize
from .npz_visualize_diff import npz_visualize_diff
from .npz_dump import npz_dump
from .npz_predict import npz_predict
import numpy as np
import sys
import struct

def bf16_to_fp32(bf16_value):
    return struct.unpack('<f', struct.pack('<HH', 0, bf16_value))[0]

def get_npz_shape(args):
    if (len(args) < 2):
        print("Usage: {} get_shape in.npz name1".format(sys.argv[0]))
        exit(-1)
    npz_in = np.load(args[0])
    shape = npz_in[args[1]].shape
    ret = ""
    for i in shape:
        if i is shape[-1]:
            ret = ret + "{}".format(i)
        else:
            ret = ret + "{},".format(i)
    print(ret)
    exit(0)

def npz_rename(args):
    if len(args) < 2:
        print("Usage: {} rename in.npz name1 name2".format(sys.argv[0]))
        exit(-1)
    npz_in = np.load(args[0])
    npz_out = {}
    d = npz_in[args[1]]
    npz_out[args[2]] = d
    np.savez(args[0], **npz_out)

def npz_extract(args):
    if len(args) < 3:
        print("Usage: {} extract in.npz out.npz arr1,arr2,arr3".format(sys.argv[0]))
        exit(-1)
    npz_in = np.load(args[0])
    npz_out = {}
    for s in args[2].split(','):
        print(s)
        d = npz_in[s]
        npz_out[s] = d
    np.savez(args[1], **npz_out)

def npz_to_bin(args):
    if len(args) < 3:
        print("Usage: {} filename.npz array_name filename.bin [int8|bf16|float32]".format(sys.argv[0]))
        exit(-1)
    npzfile = np.load(args[0])
    d = npzfile[args[1]]
    print('shape', d.shape)
    print('dtype', d.dtype)
    if len(args) == 4:
        dtype = args[3]
        if dtype == "int8":
            d = d.astype(np.int8)
        elif dtype == "bf16" or dtype == "uint16":
            d = d.astype(np.uint16)
        elif dtype == "float32":
            d = d.astype(np.float32)
        else:
            print("{}: Invalid dtype {}".format(sys.argv[0], dtype))
            exit(-1)
    d.tofile(args[2])

def npz_bf16_to_fp32(args):
    if len(args) < 2:
        print("Usage: {} bf16_to_fp32 in.npz out.npz".format(sys.argv[0]))
        exit(-1)
    npz_in = np.load(args[0])
    npz_out = {}
    for s in npz_in:
        bf16_arr = npz_in[s]
        fp32_arr = np.empty_like(bf16_arr, dtype=np.float32)

        for x, y in np.nditer([bf16_arr, fp32_arr], op_flags=['readwrite']):
          if npz_in[s].dtype == np.float32:
            y[...] = x
          else:
            y[...] = bf16_to_fp32(x)

        npz_out[s] = fp32_arr

    np.savez(args[1], **npz_out)

def npz_int8_to_fp32(args):
    # just for full-int8 not deal with mix-precision
    if len(args) < 3:
        print("Usage: {} int8_to_fp32 in.npz quantized_op_info.csv out.npz".format(sys.argv[0]))
        exit(-1)
    npz_in = np.load(args[0])
    _, _, _, thresholds = load_op_info(args[1])
    npz_out = {}
    for name in npz_in.files:
        int8_arr = npz_in[name]
        if name in thresholds and not thresholds[name] == 0.0:
            npz_out[name] = dequantize(int8_arr, thresholds[name]).astype(np.float32)
        else:
            npz_out[name] = int8_arr.astype(np.float32)
    np.savez(args[2], **npz_out)


def npz_transpose(args):
    if len(args) < 3:
        print("Usage: {} transpose in.npz nhwc nchw ".format(sys.argv[0]))
        exit(-1)
    npz_in = np.load(args[0])
    ref_data_format = args[1]
    target_data_format = args[2]
    npz_out = {}
    mapping = {ref_data_format[0] : 0,
               ref_data_format[1] : 1,
               ref_data_format[2] : 2,
               ref_data_format[3] : 3}
    tranpose = (
        mapping[target_data_format[0]],
        mapping[target_data_format[1]],
        mapping[target_data_format[2]],
        mapping[target_data_format[3]],
    )
    for k, v in npz_in.items():
        if len(v.shape) != 4:
            npz_out[k] = v
        else:
            npz_out[k] = np.ascontiguousarray(np.transpose(v, tranpose))

    np.savez(args[0], **npz_out)
    return True
