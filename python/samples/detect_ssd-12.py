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
        cv2.rectangle(img, (x0, y0), (x1, y1), color, 2)

        txt_bk_color = (_COLORS[cls_id] * 255 * 0.7).astype(np.uint8).tolist()
        cv2.rectangle(img, (x0, y0 + 1), (x0 + txt_size[0] + 1, y0 + int(1.5 * txt_size[1])),
                      txt_bk_color, -1)
        cv2.putText(img, text, (x0, y0 + txt_size[1]), font, 0.4, txt_color, thickness=1)
    return img


# This function is referenced from https://github.com/kuangliu/pytorch-ssd.
def calc_iou(box1, box2):
    """ Calculation of IoU,
        Reference to https://github.com/kuangliu/pytorch-ssd
        input:
            box1 (N, 4)
            box2 (M, 4)
        output:
            IoU (N, M)
    """
    N = box1.shape[0]
    M = box2.shape[0]
    be1 = np.tile(np.expand_dims(box1, axis=1), (1, M, 1))
    be2 = np.tile(np.expand_dims(box2, axis=0), (N, 1, 1))
    # Left Top & Right Bottom
    lt = np.maximum(be1[:, :, :2], be2[:, :, :2])
    rb = np.minimum(be1[:, :, 2:], be2[:, :, 2:])
    delta = rb - lt
    delta[delta < 0] = 0
    intersect = delta[:, :, 0] * delta[:, :, 1]
    #*mask1.float()*mask2.float()

    delta1 = be1[:, :, 2:] - be1[:, :, :2]
    area1 = delta1[:, :, 0] * delta1[:, :, 1]
    delta2 = be2[:, :, 2:] - be2[:, :, :2]
    area2 = delta2[:, :, 0] * delta2[:, :, 1]

    iou = intersect / (area1 + area2 - intersect)
    return iou


def softmax(x, dim=0):
    """Compute the softmax of vector x."""
    if dim < 0:
        dim += len(x.shape)
    x -= np.max(x, axis=dim, keepdims=True)
    exp_x = np.exp(x)
    softmax_x = exp_x / np.sum(exp_x, axis=dim, keepdims=True)
    return softmax_x


def _exponential(x):
    return np.exp(x)


# This function is referenced from https://github.com/kuangliu/pytorch-ssd.
class Encoder(object):
    """
        dboxes: num_dboxes x 4,
            encoder: input ltrb format, output xywh format
            decoder: input xywh format, output ltrb format

        encode:
            input  : bboxes_in (nboxes x 4), labels_in (nboxes)
            output : bboxes_out (num_dboxes x 4), labels_out (num_dboxes)
            iou_thres : IoU threshold of bboexes
        decode:
            input  : bboxes_in (num_dboxes x 4), scores_in (num_dboxes x nitems)
            output : bboxes_out (nboxes x 4), labels_out (nboxes)
            iou_thres : IoU threshold of bboexes
            max_output : maximum number of output bboxes
    """

    def __init__(self, dboxes):
        self.dboxes = dboxes(order="ltrb")
        self.dboxes_xywh = np.expand_dims(dboxes(order="xywh"), axis=0)
        self.nboxes = self.dboxes.shape[0]
        self.scale_xy = dboxes.scale_xy
        self.scale_wh = dboxes.scale_wh

    def encode(self, bboxes_in, labels_in, iou_thres=0.5):

        ious = calc_iou(bboxes_in, self.dboxes)
        best_dbox_ious, best_dbox_idx = ious.max(dim=0)
        best_bbox_ious, best_bbox_idx = ious.max(dim=1)

        # set best ious 2.0
        best_dbox_ious.index_fill_(0, best_bbox_idx, 2.0)

        idx = np.arange(0, best_bbox_idx.shape[0], dtype=np.int64)
        best_dbox_idx[best_bbox_idx[idx]] = idx

        # filter IoU > 0.5
        masks = best_dbox_ious > iou_thres
        labels_out = np.zeros(self.nboxes, dtype=int)
        labels_out[masks] = labels_in[best_dbox_idx[masks]]
        bboxes_out = self.dboxes.clone()
        bboxes_out[masks, :] = bboxes_in[best_dbox_idx[masks], :]
        # Transform format to xywh format
        x, y, w, h = 0.5*(bboxes_out[:, 0] + bboxes_out[:, 2]), \
                     0.5*(bboxes_out[:, 1] + bboxes_out[:, 3]), \
                     -bboxes_out[:, 0] + bboxes_out[:, 2], \
                     -bboxes_out[:, 1] + bboxes_out[:, 3]
        bboxes_out[:, 0] = x
        bboxes_out[:, 1] = y
        bboxes_out[:, 2] = w
        bboxes_out[:, 3] = h
        return bboxes_out, labels_out

    def scale_back_batch(self, bboxes_in, scores_in):
        """
            Do scale and transform from xywh to ltrb
            suppose input Nx4xnum_bbox Nxlabel_numxnum_bbox
        """
        exponential_v = np.vectorize(_exponential)

        bboxes_in = bboxes_in.transpose([0, 2, 1])
        scores_in = scores_in.transpose([0, 2, 1])

        bboxes_in[:, :, :2] = self.scale_xy * bboxes_in[:, :, :2]
        bboxes_in[:, :, 2:] = self.scale_wh * bboxes_in[:, :, 2:]

        bboxes_in[:, :, :
                  2] = bboxes_in[:, :, :2] * self.dboxes_xywh[:, :, 2:] + self.dboxes_xywh[:, :, :2]
        bboxes_in[:, :, 2:] = exponential_v(bboxes_in[:, :, 2:]) * self.dboxes_xywh[:, :, 2:]

        # Transform format to ltrb
        l, t, r, b = bboxes_in[:, :, 0] - 0.5*bboxes_in[:, :, 2],\
                     bboxes_in[:, :, 1] - 0.5*bboxes_in[:, :, 3],\
                     bboxes_in[:, :, 0] + 0.5*bboxes_in[:, :, 2],\
                     bboxes_in[:, :, 1] + 0.5*bboxes_in[:, :, 3]

        bboxes_in[:, :, 0] = l
        bboxes_in[:, :, 1] = t
        bboxes_in[:, :, 2] = r
        bboxes_in[:, :, 3] = b

        return bboxes_in, softmax(scores_in, dim=-1)

    def decode_batch(self, bboxes_in, scores_in, iou_thres=0.45, max_output=200, conf_thres=0.05):
        bboxes, probs = self.scale_back_batch(bboxes_in, scores_in)
        output = []
        for bbox, prob in zip(np.split(bboxes, bboxes.shape[0], 0),
                              np.split(probs, bboxes.shape[0], 0)):
            bbox = bbox.squeeze(0)
            prob = prob.squeeze(0)
            output.append(
                self.decode_single(bbox, prob, iou_thres, max_output, max_output, conf_thres))
        return output

    # perform nms
    def decode_single(self, bboxes_in, scores_in, iou_thres, max_output, max_num, conf_thres):
        # Reference to https://github.com/amdegroot/ssd.pytorch

        bboxes_out = []
        scores_out = []
        labels_out = []

        for i, score in enumerate(np.split(scores_in, scores_in.shape[1], 1)):
            # skip background
            if i == 0: continue

            score = score.squeeze(1)
            mask = score > conf_thres

            bboxes, score = bboxes_in[mask, :], score[mask]
            if score.shape[0] == 0: continue

            score_idx_sorted = score.argsort(axis=0)
            # select max_output indices
            score_idx_sorted = score_idx_sorted[-max_num:]
            candidates = []

            while score_idx_sorted.size > 0:
                idx = score_idx_sorted[-1].item()
                bboxes_sorted = bboxes[score_idx_sorted, :]
                bboxes_idx = np.expand_dims(bboxes[idx, :], axis=0)
                iou_sorted = calc_iou(bboxes_sorted, bboxes_idx).squeeze()
                # only need iou < iou_thres
                score_idx_sorted = score_idx_sorted[iou_sorted < iou_thres]
                candidates.append(idx)

            bboxes_out.append(bboxes[candidates, :])
            scores_out.append(score[candidates])
            labels_out.extend([i] * len(candidates))

        bboxes_out, labels_out, scores_out = np.concatenate(bboxes_out, axis=0), \
               np.array(labels_out, dtype=int), \
               np.concatenate(scores_out, axis=0)

        max_ids = scores_out.argsort(axis=0)
        max_ids = max_ids[-max_output:]
        return bboxes_out[max_ids, :], labels_out[max_ids], scores_out[max_ids]


class DefaultBoxes(object):
    def __init__(self, fig_size, feat_size, steps, scales, aspect_ratios, \
                       scale_xy=0.1, scale_wh=0.2):

        self.feat_size = feat_size
        self.fig_size_w, self.fig_size_h = fig_size

        self.scale_xy_ = scale_xy
        self.scale_wh_ = scale_wh

        # According to https://github.com/weiliu89/caffe
        self.steps_w = [st[0] for st in steps]
        self.steps_h = [st[1] for st in steps]
        self.scales = scales
        fkw = self.fig_size_w // np.array(self.steps_w)
        fkh = self.fig_size_h // np.array(self.steps_h)
        self.aspect_ratios = aspect_ratios

        self.default_boxes = []
        # size of feature and number of feature
        for idx, sfeat in enumerate(self.feat_size):
            sfeat_w, sfeat_h = sfeat
            sk1 = scales[idx][0] / self.fig_size_w
            sk2 = scales[idx + 1][1] / self.fig_size_h
            sk3 = sqrt(sk1 * sk2)
            all_sizes = [(sk1, sk1), (sk3, sk3)]
            for alpha in aspect_ratios[idx]:
                w, h = sk1 * sqrt(alpha), sk1 / sqrt(alpha)
                all_sizes.append((w, h))
                all_sizes.append((h, w))
            for w, h in all_sizes:
                for i, j in itertools.product(range(sfeat_w), range(sfeat_h)):
                    cx, cy = (j + 0.5) / fkh[idx], (i + 0.5) / fkw[idx]
                    self.default_boxes.append((cx, cy, w, h))
        self.dboxes = np.array(self.default_boxes)
        self.dboxes.clip(min=0, max=1)
        # For IoU calculation
        self.dboxes_ltrb = self.dboxes.copy()
        self.dboxes_ltrb[:, 0] = self.dboxes[:, 0] - 0.5 * self.dboxes[:, 2]
        self.dboxes_ltrb[:, 1] = self.dboxes[:, 1] - 0.5 * self.dboxes[:, 3]
        self.dboxes_ltrb[:, 2] = self.dboxes[:, 0] + 0.5 * self.dboxes[:, 2]
        self.dboxes_ltrb[:, 3] = self.dboxes[:, 1] + 0.5 * self.dboxes[:, 3]

    @property
    def scale_xy(self):
        return self.scale_xy_

    @property
    def scale_wh(self):
        return self.scale_wh_

    def __call__(self, order="ltrb"):
        if order == "ltrb": return self.dboxes_ltrb
        if order == "xywh": return self.dboxes


def create_dboxes(input_shape):
    feat_size = [[50, 50], [25, 25], [13, 13], [7, 7], [3, 3], [3, 3]]
    steps = [(int(input_shape[0] / fs[0]), int(input_shape[1] / fs[1])) for fs in feat_size]
    # use the scales here: https://github.com/amdegroot/ssd.pytorch/blob/master/data/config.py
    scales = [(int(s * input_shape[0] / 300), int(s * input_shape[1] / 300))
              for s in [21, 45, 99, 153, 207, 261, 315]]
    aspect_ratios = [[2], [2, 3], [2, 3], [2, 3], [2], [2]]
    dboxes = DefaultBoxes(input_shape, feat_size, steps, scales, aspect_ratios)
    return dboxes


def preprocess(img, input_shape):
    img = img.resize(input_shape, Image.BILINEAR)
    img_data = np.array(img)
    img_data = np.transpose(img_data, [2, 0, 1])
    img_data = np.expand_dims(img_data, 0)
    mean_vec = np.array([0.485, 0.456, 0.406])
    stddev_vec = np.array([0.229, 0.224, 0.225])
    norm_img_data = np.zeros(img_data.shape).astype('float32')
    for i in range(img_data.shape[1]):
        norm_img_data[:, i, :, :] = (img_data[:, i, :, :] / 255 - mean_vec[i]) / stddev_vec[i]
    return norm_img_data


def postprocess(outputs, input_shape, image_size, iou_thres, conf_thres):
    loc_pred, cls_pred = None, None
    for output in outputs.values():
        if output.shape[1] == 4:
            loc_pred = output
        else:
            cls_pred = output
    dboxes = create_dboxes(input_shape)
    encoder = Encoder(dboxes)

    result = encoder.decode_batch(loc_pred,
                                  cls_pred,
                                  iou_thres,
                                  max_output=200,
                                  conf_thres=conf_thres)[0]
    loc, label, prob = [r for r in result]
    label -= 1  # ignore background class
    # Scale boxes back to original image shape:
    width, height = image_size[0], image_size[1]
    image_dims = [width, height, width, height]
    loc = loc * image_dims
    return loc, label, prob


def parse_args():
    parser = argparse.ArgumentParser(description='Inference Keras SSD-12 network.')
    parser.add_argument("--model", type=str, required=True, help="Model definition file")
    parser.add_argument("--net_input_dims",
                        type=str,
                        default="1200,1200",
                        help="(h,w) of net input")
    parser.add_argument("--input", type=str, required=True, help="Input image for testing")
    parser.add_argument("--output", type=str, required=True, help="Output image after detection")
    parser.add_argument("--conf_thres", type=float, default=0.001, help="Confidence threshold")
    parser.add_argument("--iou_thres", type=float, default=0.5, help="NMS IOU threshold")
    parser.add_argument("--score_thres", type=float, default=0.6, help="Score of the result")
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    input_shape = tuple(map(int, args.net_input_dims.split(',')))
    origin_image = Image.open(args.input)
    image_data = preprocess(origin_image, input_shape)
    data = {"image": image_data}  # input name from model

    output = dict()
    boxes, categories, confidences = list(), list(), list()
    if args.model.endswith('.onnx'):
        output = onnx_inference(data, args.model, False)
        bboxes, labels, scores = output["bboxes"], output["labels"] - 1, output[
            "scores"]  # skip background class
        # Scale boxes back to original image shape:
        width, height = origin_image.size[0], origin_image.size[1]
        image_dims = [width, height, width, height]
        bboxes = bboxes * image_dims
        for idx in range(bboxes.shape[1]):
            boxes.append(list(bboxes[0, idx]))
            categories.append(labels[0, idx])
            confidences.append(scores[0, idx])
    else:
        output = dict()
        if args.model.endswith('.mlir'):
            output = mlir_inference(data, args.model, False)
        elif args.model.endswith(".bmodel"):
            output = model_inference(data, args.model)
        elif args.model.endswith(".cvimodel"):
            output = model_inference(data, args.model, False)
        else:
            raise RuntimeError("not support modle file:{}".format(args.model))

        boxes, categories, confidences = postprocess(output, input_shape, origin_image.size,
                                                     args.iou_thres, args.conf_thres)
    image = cv2.imread(args.input)
    if boxes is not None:
        fix_img = vis(image,
                      boxes,
                      confidences,
                      categories,
                      conf=args.score_thres,
                      class_names=COCO_CLASSES)
        cv2.imwrite(args.output, fix_img)
    else:
        raise RuntimeError("model:[{}] nothing detect out:{}".format(args.model, args.input))


if __name__ == '__main__':
    main()
