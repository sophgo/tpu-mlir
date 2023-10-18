#!/usr/bin/env python3
# Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
#
# TPU-MLIR is licensed under the 2-Clause BSD License except for the
# third-party components.
#
# ==============================================================================
try:
    from tpu_mlir.python import *
except ImportError:
    pass

import numpy as np
import argparse
import cv2
import linecache
from tools.model_runner import mlir_inference, model_inference, caffe_inference


def preprocess(img, input_shape):
    img = cv2.resize(img, input_shape, interpolation=cv2.INTER_LINEAR).astype(np.uint8)
    img_data = np.array(img)
    img_data = np.expand_dims(img_data, [0, 1])
    mean_vec = np.array([0])
    stddev_vec = np.array([256.0])
    norm_img_data = np.zeros(img_data.shape).astype('float32')
    for i in range(img_data.shape[1]):
        norm_img_data[:, i] = (img_data[:, i] - mean_vec[0]) / stddev_vec[0]
    return norm_img_data


def postprocess(output):
    prob = output['prob']
    number = np.argsort(-prob.flatten())[0]
    return number


def parse_args():
    parser = argparse.ArgumentParser(description='Inference LeNet network.')
    parser.add_argument("--model_def", type=str, required=True, help="Model definition file")
    parser.add_argument("--model_data", type=str, help="Caffemodel data, only for caffe model")
    parser.add_argument("--net_input_dims", type=str, default="28,28", help="(h,w) of net input")
    parser.add_argument("--input", type=str, required=True, help="Input image for testing")
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    input_shape = tuple(map(int, args.net_input_dims.split(',')))
    origin_img = cv2.imread(args.input, 0)  #GRAY HWC
    img = preprocess(origin_img, input_shape)
    data = {'data': img}  # input name from model
    output = dict()
    if args.model_def.endswith('.prototxt') and args.model_data.endswith('.caffemodel'):
        output = caffe_inference(data, args.model_def, args.model_data, False)
    elif args.model_def.endswith('.mlir'):
        output = mlir_inference(data, args.model_def, False)
    elif args.model_def.endswith(".bmodel"):
        output = model_inference(data, args.model_def)
    else:
        raise RuntimeError("not support modle file:{}".format(args.model_def))
    number = postprocess(output)

    print("The number is ", number)


if __name__ == '__main__':
    main()
