# Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
#
# TPU-MLIR is licensed under the 2-Clause BSD License except for the
# third-party components.
#
# ==============================================================================


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
    ret = "["
    dims = len(shape)
    for i in range(dims):
        ret += "{}".format(shape[i])
        if i != dims-1:
            ret += ","
    ret += "]"
    print(ret)
    exit(0)

def npz_rename(args):
    if len(args) < 2:
        print("Usage: {} rename in.npz name1 name2".format(sys.argv[0]))
        print("       Rename array name `name1` in in.npz to `name2`")
        exit(-1)
    npz_in = np.load(args[0])
    old_name = args[1]
    new_name = args[2]
    d = npz_in[old_name]
    npz_out = {}
    npz_out.update(npz_in)
    npz_out.pop(old_name)
    npz_out[new_name] = d
    np.savez(args[0], **npz_out)

def npz_extract(args):
    if len(args) < 3:
        print("Usage: {} extract in.npz out.npz arr1,arr2,arr3".format(sys.argv[0]))
        print("       Extract arrays `arr1`,`arr2`,`arr3` in in.npz and save them into output.npz")
        exit(-1)
    npz_in = np.load(args[0])
    npz_out = {}
    for s in args[2].split(','):
        print("extract", s)
        d = npz_in[s]
        npz_out[s] = d
    np.savez(args[1], **npz_out)

def npz_remove(args):
    if len(args) < 2:
        print("Usage: {} remove in.npz arr1,arr2".format(sys.argv[0]))
        print("       Remove arrays `arr1`,`arr2` in in.npz")
        exit(-1)
    file = args[0]
    npz_in = np.load(file)
    npz_out = {}
    npz_out.update(npz_in)
    for s in args[1].split(','):
        print("remove", s)
        npz_out.pop(s)
    np.savez(file, **npz_out)

def npz_merge(args):
    if len(args) < 3:
        print("Usage: {} merge a.npz b.npz output.npz".format(sys.argv[0]))
        print("       Collect arrays in a.npz and b.npz and save them into output.npz")
        exit(-1)
    num = len(args)
    npz_out = {}
    for i in range(num-1):
        data = np.load(args[i])
        npz_out.update(data)
    np.savez(args[num-1], **npz_out)

def npz_insert(args):
    if len(args) < 2:
        print("Usage: {} insert a.npz output.npz".format(sys.argv[0]))
        print("       Insert arrays in a.npz into output.npz")
        exit(-1)
    npz_out = {}
    for i in range(2):
        data = np.load(args[i])
        npz_out.update(data)
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

def npz_to_dat(args):
    if len(args) != 2:
        print("Usage: {} to_dat filename.npz filename.bin".format(sys.argv[0]))
        exit(-1)
    datas = np.load(args[0])
    with open(args[1], "wb") as f:
        for i in datas:
            f.write(datas[i].tobytes())

def npz_to_npy(args):
    if len(args) != 2:
        print("Usage: {} to_npy filename.npz name".format(sys.argv[0]))
        exit(-1)
    name = args[1]
    datas = np.load(args[0])
    if name not in datas:
        raise RuntimeError("{} not found in {}".format(name, args[0]))
    t = datas[name]
    np.save("{}.npy".format(name), t)

def npz_bf16_to_fp32(args):
    if len(args) < 2:
        print("Usage: {} bf16_to_fp32 in.npz out.npz".format(sys.argv[0]))
        exit(-1)
    npz_in = np.load(args[0])
    npz_out = {}
    for s in npz_in:
        bf16_arr = npz_in[s]
        if bf16_arr.dtype != np.uint16:
            npz_out[s] = bf16_arr
        else:
            # Using NumPy's vectorize function to apply bf16_to_fp32 to the entire array
            vectorized_func = np.vectorize(bf16_to_fp32)
            fp32_arr = vectorized_func(bf16_arr).astype(np.float32)
            npz_out[s] = fp32_arr

    np.savez(args[1], **npz_out)

def npz_fp16_to_fp32(args):
    if len(args) < 2:
        print("Usage: {} fp16_to_fp32 in.npz out.npz".format(sys.argv[0]))
        exit(-1)
    npz_in = np.load(args[0])
    npz_out = {}
    for s in npz_in:
        fp16_arr = npz_in[s]
        if fp16_arr.dtype != np.uint16:
            npz_out[s] = fp16_arr
        else:
            # Using NumPy's vectorize function to apply bf16_to_fp32 to the entire array
            value = fp16_arr.view(np.float16)
            npz_out[s] = np.float32(value)
    np.savez(args[1], **npz_out)

def npz_reshape(args):
    if len(args) < 3:
        print("Usage: {} reshape in.npz arr shape".format(sys.argv[0]))
        print("       Reshape array `arr` in in.npz as `shape`, where shape should be of form e.g.'4,5,1'")
        exit(-1)
    npz_in = np.load(args[0])
    name = args[1]
    shape = [int(x) for x in args[2].split(',')]
    d = npz_in[name].reshape(shape)
    npz_out = {}
    npz_out.update(npz_in)
    npz_out[name] = d
    np.savez(args[0], **npz_out)

def npz_permute(args):
    if len(args) < 2:
        print("Usage: {} permute in.npz order".format(sys.argv[0]))
        print("       Permute all arrays in in.npz by `order`, where order should be of form e.g.'0,2,1,3'")
        print("Usage: {} permute in.npz arr order".format(sys.argv[0]))
        print("       Permute array `arr` in in.npz by `order`, where order should be of form e.g.'0,2,1,3'")
        exit(-1)
    npz_in = np.load(args[0])
    npz_out = {}
    if len(args) == 2:
        order = [int(x) for x in args[1].split(',')]
        for k, v in npz_in.items():
            npz_out[k] = np.ascontiguousarray(np.transpose(v, order))
    else:
        name = args[1]
        order = [int(x) for x in args[2].split(',')]
        d = np.ascontiguousarray(np.transpose(npz_in[name], order))
        npz_out.update(npz_in)
        npz_out[name] = d

    np.savez(args[0], **npz_out)
