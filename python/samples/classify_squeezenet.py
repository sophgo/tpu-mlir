#!/usr/bin/env python3
# ==============================================================================
#
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

from tools.model_runner import mlir_inference, model_inference, onnx_inference, caffe_inference


def preproc(img, input_shape):
    img = cv2.resize(img, input_shape, interpolation=cv2.INTER_NEAREST).astype(np.float32)
    img_data = np.array(img)
    img_data = np.transpose(img_data, (2, 0, 1))  # HWC to CHW
    img_data = np.expand_dims(img_data, 0)
    return img_data


def parse_args():
    parser = argparse.ArgumentParser(description='Inference SequeezeNet network.')
    parser.add_argument("--model_def", type=str, required=True, help="Model definition file")
    parser.add_argument("--model_data", type=str, help="Caffemodel data, only for caffe model")
    parser.add_argument("--net_input_dims", type=str, default="227,227", help="(h,w) of net input")
    parser.add_argument("--input", type=str, required=True, help="Input image for testing")
    parser.add_argument("--output",
                        type=str,
                        required=True,
                        help="Output image after classification")
    parser.add_argument("--category_file",
                        type=str,
                        required=True,
                        help="The index file of 1000 object categories")
    args = parser.parse_args()
    return args


def postproc(output, img, category_file, top_k=5):

    prob = output['prob']
    top_k_idx = np.argsort(-prob.flatten())[:top_k]
    print("top5", top_k_idx)
    print("top5data", prob[0][top_k_idx])
    txt_bk_color = (0, 0, 0)
    txt_color = (255, 255, 255)
    font = cv2.FONT_HERSHEY_DUPLEX

    line_txt = []
    line_txt.append('Top-{:d}'.format(top_k))
    txt_size = cv2.getTextSize(line_txt[0], font, 0.4, 1)[0]
    #text left_bottom
    txt_x0 = 1
    txt_y0 = txt_size[1]
    # rectangle left top
    rect_x0 = 0
    rect_y0 = 0
    # rectangle right bottom
    rect_x1 = txt_size[0] + 2
    rect_y1 = int(1.5 * txt_size[1])
    step = int(1.5 * txt_size[1])

    img = cv2.rectangle(img, (rect_x0, rect_y0), (rect_x1, rect_y1), txt_bk_color, thickness=-1)
    img = cv2.putText(img, line_txt[0], (txt_x0, txt_y0), font, 0.4, txt_color, thickness=1)

    for i in range(top_k):
        line_txt.append(linecache.getline(category_file, top_k_idx[i] + 1).strip('\n'))
        txt_size = cv2.getTextSize(line_txt[i + 1], font, 0.4, 1)[0]
        img = cv2.rectangle(img, (0, rect_y1 + i * step),
                            (txt_size[0] + 2, rect_y1 + (i + 1) * step),
                            txt_bk_color,
                            thickness=-1)
        img = cv2.putText(img,
                          line_txt[i + 1], (1, txt_y0 + (i + 1) * step),
                          font,
                          0.4,
                          txt_color,
                          thickness=1)

    return img


def main():
    args = parse_args()
    input_shape = tuple(map(int, args.net_input_dims.split(',')))
    origin_img = cv2.imread(args.input)  #BGR HWC
    img = preproc(origin_img, input_shape)
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
    fix_img = postproc(output, origin_img, args.category_file, top_k=5)

    cv2.imwrite(args.output, fix_img)


if __name__ == '__main__':
    main()
