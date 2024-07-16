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
from tools.model_runner import mlir_inference, model_inference, onnx_inference
from PIL import Image
from math import sqrt, ceil
import itertools
import torch


def vis(img, boxes, scores, landmarks, conf=0.5):
    for i in range(len(boxes)):
        box = boxes[i]
        score = scores[i]
        if score < conf:
            continue
        x0 = int(box[0])
        y0 = int(box[1])
        x1 = int(box[2])
        y1 = int(box[3])
        text = '{:.1f}%'.format(score * 100)
        rectangle_color = (255, 128, 0)
        txt_color = (255, 255, 255)
        font = cv2.FONT_HERSHEY_SIMPLEX

        txt_size = cv2.getTextSize(text, font, 0.4, 1)[0]
        cv2.rectangle(img, (x0, y0), (x1, y1), rectangle_color, 2)

        txt_bk_color = rectangle_color
        cv2.rectangle(img, (x0, y0 + 1), (x0 + txt_size[0] + 1, y0 + int(1.5 * txt_size[1])),
                      txt_bk_color, -1)
        cv2.putText(
            img, text, (x0, y0 + txt_size[1]), font, 0.4, txt_color, thickness=1)

    point_color = (0, 255, 0)
    for landmark in landmarks:
        for mark in landmark:
            cv2.circle(img, (int(mark[0]), int(mark[1])), 1, point_color, -1)

    return img


def landmark_pred(boxes, landmark_deltas):
    if boxes.shape[0] == 0:
        return np.zeros((0, landmark_deltas.shape[1]))
    boxes = boxes.astype(np.float32, copy=False)
    widths = boxes[:, 2] - boxes[:, 0] + 1.0
    heights = boxes[:, 3] - boxes[:, 1] + 1.0
    ctr_x = boxes[:, 0] + 0.5 * (widths - 1.0)
    ctr_y = boxes[:, 1] + 0.5 * (heights - 1.0)
    pred = landmark_deltas.copy()
    for i in range(5):
        pred[:, i, 0] = landmark_deltas[:, i, 0] * widths + ctr_x
        pred[:, i, 1] = landmark_deltas[:, i, 1] * heights + ctr_y
    return pred


def clip_boxes(boxes, image_shape):
    # x1 >= 0
    boxes[:, 0::4] = np.maximum(np.minimum(
        boxes[:, 0::4], image_shape[1] - 1), 0)
    # y1 >= 0
    boxes[:, 1::4] = np.maximum(np.minimum(
        boxes[:, 1::4], image_shape[0] - 1), 0)
    # x2 < im_shape[1]
    boxes[:, 2::4] = np.maximum(np.minimum(
        boxes[:, 2::4], image_shape[1] - 1), 0)
    # y2 < im_shape[0]
    boxes[:, 3::4] = np.maximum(np.minimum(
        boxes[:, 3::4], image_shape[0] - 1), 0)
    return boxes


def bbox_pred(boxes, box_deltas):
    """
      Transform the set of class-agnostic boxes into class-specific boxes
      by applying the predicted offsets (box_deltas)
      :param boxes: !important [N, 4]
      :param box_deltas: [N, 4 * num_classes]
      :return: [N, 4 * num_classes]
    """

    if boxes.shape[0] == 0:
        return np.zeros((0, box_deltas.shape[1]))

    boxes = boxes.astype(np.float32, copy=False)
    widths = boxes[:, 2] - boxes[:, 0] + 1.0
    heights = boxes[:, 3] - boxes[:, 1] + 1.0
    ctr_x = boxes[:, 0] + 0.5 * (widths - 1.0)
    ctr_y = boxes[:, 1] + 0.5 * (heights - 1.0)

    dx = box_deltas[:, 0:1]
    dy = box_deltas[:, 1:2]
    dw = box_deltas[:, 2:3]
    dh = box_deltas[:, 3:4]

    pred_ctr_x = dx * widths[:, np.newaxis] + ctr_x[:, np.newaxis]
    pred_ctr_y = dy * heights[:, np.newaxis] + ctr_y[:, np.newaxis]
    pred_w = np.exp(dw) * widths[:, np.newaxis]
    pred_h = np.exp(dh) * heights[:, np.newaxis]

    pred_boxes = np.zeros(box_deltas.shape)
    # x1
    pred_boxes[:, 0:1] = pred_ctr_x - 0.5 * (pred_w - 1.0)
    # y1
    pred_boxes[:, 1:2] = pred_ctr_y - 0.5 * (pred_h - 1.0)
    # x2
    pred_boxes[:, 2:3] = pred_ctr_x + 0.5 * (pred_w - 1.0)
    # y2
    pred_boxes[:, 3:4] = pred_ctr_y + 0.5 * (pred_h - 1.0)

    if box_deltas.shape[1] > 4:
        pred_boxes[:, 4:] = box_deltas[:, 4:]

    return pred_boxes


def anchors_plane(height, width, stride, anchors):
    A = anchors.shape[0]
    c_0_2 = np.tile(np.arange(0, width)[
                    np.newaxis, :, np.newaxis, np.newaxis], (height, 1, A, 1))
    c_1_3 = np.tile(np.arange(0, height)[
                    :, np.newaxis, np.newaxis, np.newaxis], (1, width, A, 1))
    all_anchors = np.concatenate([c_0_2, c_1_3, c_0_2, c_1_3], axis=-1) * stride + np.tile(
        anchors[np.newaxis, np.newaxis, :, :], (height, width, 1, 1))
    return all_anchors


def nms(dets, threshold):
    'Non Maximum Suppression'
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    ndets = dets.shape[0]
    suppressed = np.zeros((ndets), dtype=np.int32)

    keep = []
    for _i in range(ndets):
        i = order[_i]
        if suppressed[i] == 1:
            continue
        keep.append(i)
        ix1 = x1[i]
        iy1 = y1[i]
        ix2 = x2[i]
        iy2 = y2[i]
        iarea = areas[i]
        for _j in range(_i + 1, ndets):
            j = order[_j]
            if suppressed[j] == 1:
                continue
            xx1 = max(ix1, x1[j])
            yy1 = max(iy1, y1[j])
            xx2 = min(ix2, x2[j])
            yy2 = min(iy2, y2[j])
            w = max(0.0, xx2 - xx1 + 1)
            h = max(0.0, yy2 - yy1 + 1)
            inter = w * h
            ovr = inter / (iarea + areas[j] - inter)
            if ovr >= threshold:
                suppressed[j] = 1

    return keep


def postprocess(image, cls_heads, box_heads, landmark_heads, image_shape, image_scale, threshold=0.5, nms_threshold=0.4):
    # Inference post-processing
    strides = [32, 16, 8]
    _anchors_fpn = {
        32: np.array([[-248., -248.,  263.,  263.], [-120., -120.,  135.,  135.]], dtype=np.float32),
        16: np.array([[-56., -56.,  71.,  71.], [-24., -24.,  39.,  39.]], dtype=np.float32),
        8: np.array([[-8., -8., 23., 23.], [0.,  0., 15., 15.]], dtype=np.float32)
    }
    num_anchors = {32: 2, 16: 2, 8: 2}
    bbox_stds = [1.0, 1.0, 1.0, 1.0]
    proposals_list = []
    scores_list = []
    landmarks_list = []

    for stride, scores, bbox_deltas, landmark_deltas in zip(strides, cls_heads, box_heads, landmark_heads):
        scores = scores[:, num_anchors[stride]:, :, :]
        height, width = bbox_deltas.shape[2], bbox_deltas.shape[3]
        A = num_anchors[stride]
        K = height * width
        anchors_fpn = _anchors_fpn[stride]
        anchors = anchors_plane(height, width, stride, anchors_fpn)
        anchors = anchors.reshape((K * A), 4)
        scores = scores.transpose((0, 2, 3, 1)).reshape((-1, 1))
        bbox_deltas = bbox_deltas.transpose((0, 2, 3, 1))
        bbox_pred_len = bbox_deltas.shape[3] // A
        bbox_deltas = bbox_deltas.reshape((-1, bbox_pred_len))
        bbox_deltas[:, 0::4] = bbox_deltas[:, 0::4] * bbox_stds[0]
        bbox_deltas[:, 1::4] = bbox_deltas[:, 1::4] * bbox_stds[1]
        bbox_deltas[:, 2::4] = bbox_deltas[:, 2::4] * bbox_stds[2]
        bbox_deltas[:, 3::4] = bbox_deltas[:, 3::4] * bbox_stds[3]
        proposals = bbox_pred(anchors, bbox_deltas)
        proposals = clip_boxes(
            proposals, image_shape[:2])
        scores_ravel = scores.ravel()
        order = np.where(scores_ravel >= threshold)[0]
        proposals = proposals[order, :]
        scores = scores[order]

        proposals[:, 0] /= image_scale[1]
        proposals[:, 1] /= image_scale[0]
        proposals[:, 2] /= image_scale[1]
        proposals[:, 3] /= image_scale[0]

        proposals_list.append(proposals)
        scores_list.append(scores)

        landmark_pred_len = landmark_deltas.shape[1] // A
        landmark_deltas = landmark_deltas.transpose(
            (0, 2, 3, 1)).reshape((-1, 5, landmark_pred_len // 5))
        landmarks = landmark_pred(anchors, landmark_deltas)
        landmarks = landmarks[order, :]
        landmarks[:, :, 0] /= image_scale[1]
        landmarks[:, :, 1] /= image_scale[0]
        landmarks_list.append(landmarks)

    proposals = np.vstack(proposals_list)
    landmarks = None
    if proposals.shape[0] == 0:
        landmarks = np.zeros((0, 5, 2))
        return np.zeros((0, 1)), np.zeros((0, 4)), landmarks
    scores = np.vstack(scores_list)
    scores_ravel = scores.ravel()
    order = scores_ravel.argsort()[::-1]
    proposals = proposals[order, :]
    scores = scores[order]

    landmarks = np.vstack(landmarks_list)
    landmarks = landmarks[order].astype(np.float32, copy=False)

    pre_det = np.hstack((proposals[:, 0:4], scores)).astype(
        np.float32, copy=False)
    keep = nms(pre_det, nms_threshold)
    det = np.hstack((pre_det, proposals[:, 4:]))
    det = det[keep, :]
    landmarks = landmarks[keep]

    return (det[:, 4], det[:, 0:4], landmarks)


def resize_image(img, scales):
    img_h, img_w = img.shape[0:2]
    target_h = scales[0]
    target_w = scales[1]
    im_scale = [target_h / float(img_h), target_w / float(img_w)]

    if im_scale != 1.0:
        img = cv2.resize(
            img,
            None,
            None,
            fx=im_scale[1],
            fy=im_scale[0],
            interpolation=cv2.INTER_LINEAR
        )

    return img, im_scale


def preprocess(img, input_shape):
    pixel_means = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    pixel_stds = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    pixel_scale = float(1.0)

    img, im_scale = resize_image(img, input_shape)
    img = img.astype(np.float32)
    im_tensor = np.zeros(
        (1, img.shape[0], img.shape[1], img.shape[2]), dtype=np.float32)

    # Make image scaling + BGR2RGB conversion + transpose (N,H,W,C) to (N,C,H,W)
    for i in range(3):
        im_tensor[0, :, :, i] = (
            img[:, :, 2 - i] / pixel_scale - pixel_means[2 - i]) / pixel_stds[2 - i]
    im_tensor = im_tensor.transpose([0, 3, 1, 2])
    return im_tensor, img.shape[0:2], im_scale


def parse_args():
    parser = argparse.ArgumentParser(
        description='Inference retinaface network.')
    parser.add_argument("--model", type=str, required=True,
                        help="Model definition file")
    parser.add_argument("--net_input_dims", type=str,
                        default="240,320", help="(h,w) of net input")
    parser.add_argument("--input", type=str, required=True,
                        help="Input image for testing")
    parser.add_argument("--output",
                        type=str,
                        required=True,
                        help="Output image after detection")
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    origin_image = cv2.imread(args.input)
    input_shape = tuple(map(int, args.net_input_dims.split(',')))
    image_data, image_shape, image_scale = preprocess(
        origin_image, input_shape)
    data = {"data": image_data}  # input name from model

    output = dict()

    cls_names = ['face_rpn_cls_prob_reshape_stride32',
                 'face_rpn_cls_prob_reshape_stride16', 'face_rpn_cls_prob_reshape_stride8']
    bbx_names = ['face_rpn_bbox_pred_stride32',
                 'face_rpn_bbox_pred_stride16', 'face_rpn_bbox_pred_stride8']
    landmark_names = ['face_rpn_landmark_pred_stride32',
                      'face_rpn_landmark_pred_stride16', 'face_rpn_landmark_pred_stride8']
    cls_heads = []
    bbx_heads = []
    landmark_heads = []
    if args.model.endswith('.onnx'):
        outputs = onnx_inference(data, args.model, False)
    else:
        if args.model.endswith('.mlir'):
            outputs = mlir_inference(data, args.model, False)
        elif args.model.endswith(".bmodel"):
            outputs = model_inference(data, args.model)
        elif args.model.endswith(".cvimodel"):
            outputs = model_inference(data, args.model, False)
        else:
            raise RuntimeError("not support modle file:{}".format(args.model))

    output_keys = set(outputs.keys())

    for bbx_name, cls_name, landmark_name in zip(bbx_names, cls_names, landmark_names):
        for key in output_keys:
            if key.startswith(cls_name):
                cls_heads.append(outputs[key])
            elif key.startswith(bbx_name):
                bbx_heads.append(outputs[key])
            elif key.startswith(landmark_name):
                landmark_heads.append(outputs[key])
            else:
                continue

    probs, boxes, landmarks = postprocess(
        image_data, cls_heads, bbx_heads, landmark_heads, image_shape, image_scale)

    if boxes.shape[0] > 0:
        fix_img = vis(origin_image, boxes, probs, landmarks)
        cv2.imwrite(args.output, fix_img)
    else:
        raise RuntimeError(
            "model:[{}] nothing detect out:{}".format(args.model, args.input))


if __name__ == '__main__':
    main()
