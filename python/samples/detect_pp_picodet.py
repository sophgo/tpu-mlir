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
import os
import sys
import argparse
import cv2
import math
from tools.model_runner import mlir_inference, model_inference, onnx_inference
from utils.preprocess import supported_customization_format

COCO_CLASSES = ("person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
                "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
                "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
                "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis",
                "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
                "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife",
                "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot",
                "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed",
                "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard",
                "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book",
                "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush")

_COLORS = np.array([
    0.000, 0.447, 0.741, 0.850, 0.325, 0.098, 0.929, 0.694, 0.125, 0.494, 0.184, 0.556, 0.466,
    0.674, 0.188, 0.301, 0.745, 0.933, 0.635, 0.078, 0.184, 0.300, 0.300, 0.300, 0.600, 0.600,
    0.600, 1.000, 0.000, 0.000, 1.000, 0.500, 0.000, 0.749, 0.749, 0.000, 0.000, 1.000, 0.000,
    0.000, 0.000, 1.000, 0.667, 0.000, 1.000, 0.333, 0.333, 0.000, 0.333, 0.667, 0.000, 0.333,
    1.000, 0.000, 0.667, 0.333, 0.000, 0.667, 0.667, 0.000, 0.667, 1.000, 0.000, 1.000, 0.333,
    0.000, 1.000, 0.667, 0.000, 1.000, 1.000, 0.000, 0.000, 0.333, 0.500, 0.000, 0.667, 0.500,
    0.000, 1.000, 0.500, 0.333, 0.000, 0.500, 0.333, 0.333, 0.500, 0.333, 0.667, 0.500, 0.333,
    1.000, 0.500, 0.667, 0.000, 0.500, 0.667, 0.333, 0.500, 0.667, 0.667, 0.500, 0.667, 1.000,
    0.500, 1.000, 0.000, 0.500, 1.000, 0.333, 0.500, 1.000, 0.667, 0.500, 1.000, 1.000, 0.500,
    0.000, 0.333, 1.000, 0.000, 0.667, 1.000, 0.000, 1.000, 1.000, 0.333, 0.000, 1.000, 0.333,
    0.333, 1.000, 0.333, 0.667, 1.000, 0.333, 1.000, 1.000, 0.667, 0.000, 1.000, 0.667, 0.333,
    1.000, 0.667, 0.667, 1.000, 0.667, 1.000, 1.000, 1.000, 0.000, 1.000, 1.000, 0.333, 1.000,
    1.000, 0.667, 1.000, 0.333, 0.000, 0.000, 0.500, 0.000, 0.000, 0.667, 0.000, 0.000, 0.833,
    0.000, 0.000, 1.000, 0.000, 0.000, 0.000, 0.167, 0.000, 0.000, 0.333, 0.000, 0.000, 0.500,
    0.000, 0.000, 0.667, 0.000, 0.000, 0.833, 0.000, 0.000, 1.000, 0.000, 0.000, 0.000, 0.167,
    0.000, 0.000, 0.333, 0.000, 0.000, 0.500, 0.000, 0.000, 0.667, 0.000, 0.000, 0.833, 0.000,
    0.000, 1.000, 0.000, 0.000, 0.000, 0.143, 0.143, 0.143, 0.286, 0.286, 0.286, 0.429, 0.429,
    0.429, 0.571, 0.571, 0.571, 0.714, 0.714, 0.714, 0.857, 0.857, 0.857, 0.000, 0.447, 0.741,
    0.314, 0.717, 0.741, 0.50, 0.5, 0
]).astype(np.float32).reshape(-1, 3)

COCO_IDX = [
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 27, 28,
    31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55,
    56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84,
    85, 86, 87, 88, 89, 90
]

ANCHORS = {  # stride: anchor
    8: [[1.25000, 1.62500], [2.00000, 3.75000], [4.12500, 2.87500]],
    16: [[1.87500, 3.81250], [3.87500, 2.81250], [3.68750, 7.43750]],
    32: [[3.62500, 2.81250], [4.87500, 6.18750], [11.65625, 10.18750]]
}

customization_format_attributes = {
    'RGB_PLANAR': ('rgb', 'nchw'),
    'RGB_PACKED': ('rgb', 'nhwc'),
    'BGR_PLANAR': ('bgr', 'nchw'),
    'BGR_PACKED': ('bgr', 'nhwc'),
    'GRAYSCALE': ('gray', 'nchw'),
    'YUV420_PLANAR': ('bgr', 'nchw'),
    'YUV_NV12': ('bgr', 'nchw'),
    'YUV_NV21': ('bgr', 'nchw'),
    'RGBA_PLANAR': ('rgba', 'nchw')
}


def vis(img, boxes, scores, cls_ids, conf=0.5, class_names=None):

    for i in range(len(boxes)):
        box = boxes[i]
        cls_id = int(cls_ids[i])
        score = scores[i]
        if score < conf:
            continue
        x0 = int(box[0])
        y0 = int(box[1])
        x1 = int(box[2])
        y1 = int(box[3])
        color = (_COLORS[cls_id] * 255).astype(np.uint8).tolist()
        text = '{}:{:.1f}%'.format(class_names[cls_id], score * 100)
        txt_color = (0, 0, 0) if np.mean(_COLORS[cls_id]) > 0.5 else (255, 255, 255)
        font = cv2.FONT_HERSHEY_SIMPLEX

        txt_size = cv2.getTextSize(text, font, 0.4, 1)[0]
        cv2.rectangle(img, (x0, y0), (x1, y1), color, 1)

        txt_bk_color = (_COLORS[cls_id] * 255 * 0.7).astype(np.uint8).tolist()
        cv2.rectangle(img, (x0, y0 + 1), (x0 + txt_size[0] + 1, y0 + int(1.5 * txt_size[1])),
                      txt_bk_color, -1)
        cv2.putText(img, text, (x0, y0 + txt_size[1]), font, 0.4, txt_color, thickness=1)

    return img


def preproc(img, input_size, pixel_format, channel_format, fuse_pre, swap=(2, 0, 1)):
    if len(img.shape) == 3:
        padded_img = np.ones((input_size[0], input_size[1], 3), dtype=np.uint8) * 114  # 114
    else:
        padded_img = np.ones(input_size, dtype=np.uint8) * 114  # 114

    r = min(input_size[0] / img.shape[0], input_size[1] / img.shape[1])
    resized_img = cv2.resize(
        img,
        (int(img.shape[1] * r), int(img.shape[0] * r)),
        interpolation=cv2.INTER_LINEAR,
    ).astype(np.uint8)
    top = int((input_size[0] - int(img.shape[0] * r)) / 2)
    left = int((input_size[1] - int(img.shape[1] * r)) / 2)
    padded_img[top:int(img.shape[0] * r) + top, left:int(img.shape[1] * r) + left] = resized_img

    if (channel_format == 'nchw'):
        padded_img = padded_img.transpose(swap)  # HWC to CHW
    if (pixel_format == 'rgb'):
        padded_img = padded_img[::-1]  # BGR to RGB

    padded_img = np.ascontiguousarray(padded_img, dtype=np.float32 if not fuse_pre else np.uint8)

    return padded_img, r, top, left


def _sigmoid(x):
    return 1. / (1. + np.exp(-x))


def make_grid_v2(featmap_size, stride):
    feat_h, feat_w = featmap_size
    shift_x = np.arange(0, feat_w) * stride
    shift_y = np.arange(0, feat_h) * stride
    xv, yv = np.meshgrid(shift_x, shift_y)
    xv = xv.flatten()
    yv = yv.flatten()
    cx = xv + 0.5 * (stride - 1)
    cy = yv + 0.5 * (stride - 1)
    return np.stack((cx, cy), axis=-1)


def postproc_pico(outputs, input_shape, top, left, anchors=ANCHORS):
    outputs = list(outputs.values())
    num_outs = 4
    cls_scores, bbox_preds = outputs[:num_outs], outputs[num_outs:]
    det_bboxes, det_conf, det_classid = get_bboxes_single(cls_scores,
                                                          bbox_preds,
                                                          1,
                                                          input_shape,
                                                          rescale=False)
    det_bboxes[:, 0] -= left
    det_bboxes[:, 2] -= left
    det_bboxes[:, 1] -= top
    det_bboxes[:, 3] -= top

    return det_conf, det_bboxes, det_classid


def softmax(x, axis=1):
    x_exp = np.exp(x)
    x_sum = np.sum(x_exp, axis=axis, keepdims=True)
    s = x_exp / x_sum
    return s


def distance2bbox(points, distance, max_shape=None):
    x1 = points[:, 0] - distance[:, 0]
    y1 = points[:, 1] - distance[:, 1]
    x2 = points[:, 0] + distance[:, 2]
    y2 = points[:, 1] + distance[:, 3]
    if max_shape is not None:
        x1 = np.clip(x1, 0, max_shape[1])
        y1 = np.clip(y1, 0, max_shape[0])
        x2 = np.clip(x2, 0, max_shape[1])
        y2 = np.clip(y2, 0, max_shape[0])
    return np.stack([x1, y1, x2, y2], axis=-1)


def get_bboxes_single(cls_scores, bbox_preds, scale_factor, input_shape, rescale=False):
    mlvl_bboxes = []
    mlvl_scores = []
    mlvl_anchors = []
    strides = [8, 16, 32, 64]
    project = np.array([0, 1, 2, 3, 4, 5, 6, 7])
    for i in range(len(strides)):
        anchors = make_grid_v2(
            (math.ceil(input_shape[0] / strides[i]), math.ceil(input_shape[1] / strides[i])),
            strides[i])
        mlvl_anchors.append(anchors)
    for stride, cls_score, bbox_pred, anchors in zip(strides, cls_scores, bbox_preds, mlvl_anchors):
        if cls_score.ndim == 3:
            cls_score = cls_score.squeeze(axis=0)
        if bbox_pred.ndim == 3:
            bbox_pred = bbox_pred.squeeze(axis=0)
        bbox_pred = softmax(bbox_pred.reshape(-1, 8), axis=1)
        bbox_pred = np.dot(bbox_pred, project).reshape(-1, 4)
        bbox_pred *= stride
        nms_pre = 1000
        if nms_pre > 0 and cls_score.shape[0] > nms_pre:
            max_scores = cls_score.max(axis=1)
            topk_inds = max_scores.argsort()[::-1][0:nms_pre]
            anchors = anchors[topk_inds, :]
            bbox_pred = bbox_pred[topk_inds, :]
            cls_score = cls_score[topk_inds, :]

        bboxes = distance2bbox(anchors, bbox_pred, max_shape=input_shape)
        mlvl_bboxes.append(bboxes)
        mlvl_scores.append(cls_score)

    mlvl_bboxes = np.concatenate(mlvl_bboxes, axis=0)
    if rescale:
        mlvl_bboxes /= scale_factor
    mlvl_scores = np.concatenate(mlvl_scores, axis=0)

    bboxes_wh = mlvl_bboxes.copy()
    bboxes_wh[:, 2:4] = bboxes_wh[:, 2:4] - bboxes_wh[:, 0:2]
    classIds = np.argmax(mlvl_scores, axis=1)
    confidences = np.max(mlvl_scores, axis=1)

    indices = cv2.dnn.NMSBoxes(bboxes_wh.tolist(), confidences.tolist(), 0.5, 0.5)
    if len(indices) > 0:
        indices = indices.flatten()
        mlvl_bboxes = mlvl_bboxes[indices]
        confidences = confidences[indices]
        classIds = classIds[indices]
        return mlvl_bboxes, confidences, classIds
    else:
        print('nothing detect')
        return np.array([]), np.array([]), np.array([])




def parse_args():
    parser = argparse.ArgumentParser(description='Inference Yolo v5 network.')
    parser.add_argument("--model", type=str, required=True, help="Model definition file")
    parser.add_argument("--net_input_dims", type=str, default="320,320", help="(h,w) of net input")
    parser.add_argument("--input", type=str, required=True, help="Input image for testing")
    parser.add_argument("--output", type=str, required=True, help="Output image after detection")
    parser.add_argument("--input_names", type=str, default="image", help="Input name for testing")
    parser.add_argument("--conf_thres", type=float, default=0.001, help="Confidence threshold")
    parser.add_argument("--iou_thres", type=float, default=0.6, help="NMS IOU threshold")
    parser.add_argument("--score_thres", type=float, default=0.5, help="Score of the result")
    parser.add_argument("--fuse_preprocess",
                        action='store_true',
                        help="if the model fused prerpocess")
    parser.add_argument("--customization_format",
                        default='',
                        type=str.upper,
                        choices=supported_customization_format,
                        help="pixel and channel format of original input data")
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    input_shape = tuple(map(int, args.net_input_dims.split(',')))
    fuse_pre = args.fuse_preprocess
    customization_format = args.customization_format
    pixel_format = 'rgb'
    channel_format = 'nchw'
    if (fuse_pre and customization_format):
        pixel_format = customization_format_attributes[customization_format][0]
        channel_format = customization_format_attributes[customization_format][1]

    origin_img = cv2.imread(args.input)
    img, ratio, top, left = preproc(origin_img, input_shape, pixel_format, channel_format, fuse_pre)
    img = np.expand_dims(img, axis=0)
    if (not fuse_pre):
        img /= 255.  # 0 - 255 to 0.0 - 1.0
    data = {args.input_names: img}  # input name from model
    output = dict()
    if args.model.endswith('.onnx'):
        output = onnx_inference(data, args.model, False)
    elif args.model.endswith('.mlir'):
        output = mlir_inference(data, args.model, False)
    elif args.model.endswith(".bmodel") or args.model.endswith(".cvimodel"):
        if args.model.endswith(".cvimodel"):
            raise RuntimeError("not support cvimodel now.")
        output = model_inference(data, args.model)
    else:
        raise RuntimeError("not support modle file:{}".format(args.model))
    out = postproc_pico(output, input_shape, top, left)
    if out:
        final_scores, final_boxes, final_cls_inds = out
        final_boxes /= ratio
        fix_img = vis(origin_img,
                      final_boxes,
                      final_scores,
                      final_cls_inds,
                      conf=args.score_thres,
                      class_names=COCO_CLASSES)
        cv2.imwrite(args.output, fix_img)
    else:
        raise RuntimeError("model:[{}] nothing detect out:{}".format(args.model, args.input))


if __name__ == '__main__':
    main()
