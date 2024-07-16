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
from pathlib import Path

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

ANCHORS = [[10, 14], [23, 27], [37, 58], [81, 82], [135, 169], [344, 319]]

MASKS = [[3, 4, 5], [1, 2, 3]]


def vis(img, boxes, scores, cls_ids, conf=0.5, class_names=None):
    res_line = ''
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
        res_line += f'{text}, '
        print(text)
        txt_color = (0, 0, 0) if np.mean(_COLORS[cls_id]) > 0.5 else (255, 255, 255)
        font = cv2.FONT_HERSHEY_SIMPLEX

        txt_size = cv2.getTextSize(text, font, 0.4, 1)[0]
        cv2.rectangle(img, (x0, y0), (x1, y1), color, 2)

        txt_bk_color = (_COLORS[cls_id] * 255 * 0.7).astype(np.uint8).tolist()
        cv2.rectangle(img, (x0, y0 + 1), (x0 + txt_size[0] + 1, y0 + int(1.5 * txt_size[1])),
                      txt_bk_color, -1)
        cv2.putText(img, text, (x0, y0 + txt_size[1]), font, 0.4, txt_color, thickness=1)

    return img, res_line[:-2]


def vis2(img, dets, input_shape, conf=0.5, class_names=None):
    shape = dets.shape
    num = shape[-2]
    img_h, img_w = img.shape[0:2]
    for i in range(num):
        d = dets[0][0][i]
        cls_id = int(d[1])
        score = d[2]
        if score < conf:
            continue
        x, y, w, h = d[3] * input_shape[1], d[4] * input_shape[0], d[5] * input_shape[1], d[
            6] * input_shape[0]

        r = min(input_shape[0] / img.shape[0], input_shape[1] / img.shape[1])
        new_img_h = img_h * r
        new_img_w = img_w * r
        x -= ((input_shape[1] - new_img_w) / 2)
        y -= ((input_shape[0] - new_img_h) / 2)
        x /= r
        y /= r
        w /= r
        h /= r
        x0 = int(x - w / 2)
        y0 = int(y - h / 2)
        x1 = int(x + w / 2)
        y1 = int(y + h / 2)
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


def nms(boxes, box_confidences, iou_thres, conf_thres):
    x_coord = boxes[:, 0]
    y_coord = boxes[:, 1]
    width = boxes[:, 2]
    height = boxes[:, 3]

    areas = width * height
    ordered = box_confidences.argsort()[::-1]

    keep = list()
    while ordered.size > 0:
        i = ordered[0]
        keep.append(i)
        xx1 = np.maximum(x_coord[i], x_coord[ordered[1:]])
        yy1 = np.maximum(y_coord[i], y_coord[ordered[1:]])
        xx2 = np.minimum(x_coord[i] + width[i], x_coord[ordered[1:]] + width[ordered[1:]])
        yy2 = np.minimum(y_coord[i] + height[i], y_coord[ordered[1:]] + height[ordered[1:]])

        width1 = np.maximum(0.0, xx2 - xx1 + 1)
        height1 = np.maximum(0.0, yy2 - yy1 + 1)
        intersection = width1 * height1
        union = (areas[i] + areas[ordered[1:]] - intersection)

        iou = intersection / union

        indexes = np.where(iou <= iou_thres)[0]
        ordered = ordered[indexes + 1]

    keep = np.array(keep)
    return keep


def letterbox_image(image, size):
    '''resize image with unchanged aspect ratio using padding'''
    iw, ih = image.size
    w, h = size
    scale = min(w / iw, h / ih)
    nw = int(iw * scale)
    nh = int(ih * scale)

    image = image.resize((nw, nh), Image.Resampling.BICUBIC)
    new_image = Image.new('RGB', size, (128, 128, 128))
    new_image.paste(image, ((w - nw) // 2, (h - nh) // 2))
    return new_image


def preprocess(img, input_shape, fuse_pre=False):
    boxed_image = letterbox_image(img, tuple(reversed(input_shape)))
    image_data = np.array(boxed_image, dtype=np.float32 if not fuse_pre else np.uint8)
    image_data = np.transpose(image_data, [2, 0, 1])
    image_data = np.expand_dims(image_data, 0)
    if not fuse_pre:
        image_data /= 255.

    return image_data


def make_grid(nx, ny, stride, anchor):
    stride = np.array(stride)
    anchor = np.array(anchor)
    xv, yv = np.meshgrid(np.arange(nx), np.arange(ny))
    grid = np.stack((xv, yv), -1)
    anchor_grid = (anchor * stride).reshape(1, len(anchor), 1, 1, 2)
    return grid, anchor_grid


def process_feats(output_reshaped, mask, input_shape, image_shape):
    sigmoid_v = np.vectorize(_sigmoid)
    exponential_v = np.vectorize(_exponential)

    grid_h, grid_w, _, _ = output_reshaped.shape
    anchors = [ANCHORS[i] for i in mask]

    # Reshape to N, height, width, num_anchors, box_params:
    anchors_tensor = np.reshape(anchors, [1, 1, len(anchors), 2])

    box_xy = sigmoid_v(output_reshaped[..., :2])
    box_wh = exponential_v(output_reshaped[..., 2:4]) * anchors_tensor
    box_class_probs = sigmoid_v(output_reshaped[..., 5:])
    box_confidence = sigmoid_v(output_reshaped[..., 4])
    box_confidence = np.expand_dims(box_confidence, axis=-1)

    col = np.tile(np.arange(0, grid_w), grid_h).reshape(-1, grid_w)
    row = np.tile(np.arange(0, grid_h).reshape(-1, 1), grid_w)

    col = col.reshape(grid_h, grid_w, 1, 1).repeat(3, axis=-2)
    row = row.reshape(grid_h, grid_w, 1, 1).repeat(3, axis=-2)
    grid = np.concatenate((col, row), axis=-1)

    box_xy += grid
    box_xy /= (grid_w, grid_h)
    box_wh /= tuple(reversed(input_shape))

    input_shape = np.array(input_shape)
    image_shape = np.array(image_shape)
    new_shape = np.round(image_shape * np.min(input_shape / image_shape))
    offset = (input_shape - new_shape) / 2. / input_shape
    scale = input_shape / new_shape
    box_xy = (box_xy - offset) * scale
    box_wh *= scale

    box_xy_min = box_xy - (box_wh / 2.)
    box_xy_max = box_xy + (box_wh / 2.)
    boxes = np.concatenate((box_xy_min, box_xy_max), axis=-1)

    # boxes: centroids, box_confidence: confidence level, box_class_probs:
    # class confidence
    return boxes, box_confidence, box_class_probs


def filter_boxes(boxes, box_confidences, box_class_probs, score_thres):
    box_scores = box_confidences * box_class_probs
    box_classes = np.argmax(box_scores, axis=-1)
    box_class_scores = np.max(box_scores, axis=-1)
    pos = np.where(box_class_scores >= score_thres)

    boxes = boxes[pos]
    classes = box_classes[pos]
    scores = box_class_scores[pos]
    return boxes, classes, scores


def _exponential(x):
    return np.exp(x)


def _sigmoid(x):
    return 1. / (1. + np.exp(-x))


def postprocess(outputs,
                input_shape,
                img_shape,
                score_thres,
                iou_thres,
                conf_thres,
                anchors=ANCHORS,
                max_boxes=20):
    outputs_reshaped = []
    for output in outputs.values():
        output = output.transpose([0, 2, 3, 1])
        _, height, width, channels = output.shape
        dim1, dim2 = height, width
        dim3 = 3
        # There are CATEGORY_NUM=80 object categories
        # Outputs are reshaped as H x W x 3 x num_class
        dim4 = int(channels / dim3)
        outputs_reshaped.append(np.reshape(output, (dim1, dim2, dim3, dim4)))

    boxes, categories, confidences = list(), list(), list()
    for output, mask in zip(outputs_reshaped, MASKS):
        box, category, confidence = process_feats(output, mask, input_shape, img_shape)
        box, category, confidence = filter_boxes(box, category, confidence, score_thres)
        boxes.append(box)
        categories.append(category)
        confidences.append(confidence)

    boxes = np.concatenate(boxes)
    categories = np.concatenate(categories)
    confidences = np.concatenate(confidences)

    # Scale boxes back to original image shape:
    width, height = img_shape
    image_dims = [width, height, width, height]
    boxes = boxes * image_dims

    # Using the candidates from the previous (loop) step, we apply the non-max suppression
    # algorithm that clusters adjacent bounding boxes to a single bounding box:
    nms_boxes, nms_categories, nscores = list(), list(), list()
    for category in set(categories):
        idxs = np.where(categories == category)
        box = boxes[idxs]
        category = categories[idxs]
        confidence = confidences[idxs]

        keep = nms(box, confidence, iou_thres, conf_thres)

        nms_boxes.append(box[keep])
        nms_categories.append(category[keep])
        nscores.append(confidence[keep])

    if not nms_categories and not nscores:
        return None, None, None

    boxes = np.concatenate(nms_boxes)
    categories = np.concatenate(nms_categories)
    confidences = np.concatenate(nscores)
    return boxes, categories, confidences



def parse_args():
    # yapf: disable
    parser = argparse.ArgumentParser(description='Inference Keras Yolo3 network.')
    parser.add_argument("--model", type=str, required=True, help="Model definition file")
    parser.add_argument("--net_input_dims", type=str, default="416,416", help="(h,w) of net input")
    parser.add_argument("--input", type=str, required=True, help="Input image or dir for testing")
    parser.add_argument("--output", type=str, required=True, help="Output image or dir after detection")
    parser.add_argument("--conf_thres", type=float, default=0.001, help="Confidence threshold")
    parser.add_argument("--iou_thres", type=float, default=0.5, help="NMS IOU threshold")
    parser.add_argument("--score_thres", type=float, default=0.56, help="Score of the result")
    parser.add_argument("--fuse_preprocess", action='store_true', help="if the model fused preprocess")
    parser.add_argument("--fuse_postprocess", action='store_true', help="if the model fused postprocess")
    # yapf: enable
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    input_shape = tuple(map(int, args.net_input_dims.split(',')))

    image_datas, file_list = [], []
    if Path(args.input).is_file():
        origin_image = Image.open(args.input)
        image_data = preprocess(origin_image, input_shape, args.fuse_preprocess)
        image_datas.append(image_data)
        file_list.append(args.input)
    else:
        dir_or_files = os.listdir(args.input)
        for dir_file in dir_or_files:
            dir_file_path = os.path.join(args.input, dir_file)
            ext_name = os.path.splitext(dir_file)[-1]
            if not os.path.isdir(dir_file_path) and ext_name in [
                    '.jpg', '.png', '.jpeg', '.bmp', '.JPEG', '.JPG', '.BMP'
            ]:
                file_list.append(dir_file_path)
                origin_image = Image.open(dir_file_path)
                image_data = preprocess(origin_image, input_shape, args.fuse_preprocess)
                image_datas.append(image_data)

    fname = args.model
    fname = fname.split('/')[-1]
    file = open(f'{fname}_image_dir_result', 'w')
    for file_name, image_data in zip(file_list, image_datas):
        image_size = np.array([origin_image.size[1], origin_image.size[0]],
                              dtype=np.float32).reshape(1, 2)
        data = {}
        data["input_1"] = image_data # input name from the model

        output = dict()
        boxes, confidences, categories, dets = [], [], [], []
        if args.model.endswith('.onnx'):
            data["image_shape"] = image_size  # onnx model needs 2 inputs
            output = onnx_inference(data, args.model, False)
            all_boxes, scores, indices = None, None, None
            for out in output.values():
                if out.shape[-1] == 4:
                    all_boxes = out
                elif out.shape[1] == 80:
                    scores = out
                else:
                    indices = out
            # rank of indices in yolov3_tiny is 3
            indices = indices[0] if len(indices.shape) == 3 else indices
            for idx_ in indices:
                idx_ = tuple(map(int, idx_))
                categories.append(idx_[1])
                confidences.append(scores[idx_])
                idx_1 = (idx_[0], idx_[2])
                y0, x0, y1, x1 = all_boxes[idx_1]
                boxes.append([x0, y0, x1, y1])
        else:
            if args.model.endswith('.mlir'):
                output = mlir_inference(data, args.model, False)
            elif args.model.endswith(".bmodel"):
                output = model_inference(data, args.model)
            elif args.model.endswith(".cvimodel"):
                output = model_inference(data, args.model, False)
            else:
                raise RuntimeError("not support modle file:{}".format(args.model))
            if not args.fuse_postprocess:
                boxes, categories, confidences = postprocess(output, input_shape, origin_image.size,
                                                             args.score_thres, args.iou_thres,
                                                             args.conf_thres)
            else:
                dets = output['yolo_post']

        image = cv2.imread(file_name)
        res_line = ''
        tmpstr = file_name.split('/')[-1].replace(' ', '_')
        if boxes is not None:
            if not args.fuse_postprocess:
                fix_img, res_line = vis(image,
                                        boxes,
                                        confidences,
                                        categories,
                                        conf=args.score_thres,
                                        class_names=COCO_CLASSES)
            else:
                fix_img = vis2(image,
                               dets,
                               input_shape,
                               conf=args.score_thres,
                               class_names=COCO_CLASSES)

            if Path(args.input).is_file():
                output_file = args.output
            else:
                output_file = os.path.join(args.output, f'{tmpstr}_res')
            cv2.imwrite(output_file, fix_img)
        else:
            print('No object was detected')
        file.write(tmpstr + ': ' + res_line + '\n')
    file.close()


if __name__ == '__main__':
    main()
