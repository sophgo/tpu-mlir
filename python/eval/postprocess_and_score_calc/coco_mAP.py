# ==============================================================================
#
# Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
#
# TPU-MLIR is licensed under the 2-Clause BSD License except for the
# third-party components.
#
# ==============================================================================


import os
import cv2
import json
from .base_class import *
import numpy as np
import argparse
from pathlib import Path
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from utils.log_setting import setup_logger
logger = setup_logger('root', log_level="INFO")

class score_Parser():
    def __init__(self):
        parser = argparse.ArgumentParser(description='Eval YOLO networks.')
        parser.add_argument("--draw_image_count", type=int, default=0,
                            help="the number of images will draw results")
        parser.add_argument("--nms_score_thr", type=float, default=0.01,
                            help="Object confidence threshold")
        parser.add_argument("--nms_threshold", type=float, default=0.6,
                            help="NMS IOU threshold")
        parser.add_argument("--coco_annotation", type=str, default='/work/dataset/coco/annotations/instances_val2017.json',
                            help="annotations file")
        parser.add_argument("--coco_result_jason_file", type=str, default="restult_yolox_s.json",
                            help="Result json file")
        parser.add_argument("-s", "--score_thr", type=float, default=0.3,
                            help="Score threshould to filter the result.", )
        self.parser = parser

COCO_CLASSES = (
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
)

_COLORS = np.array(
    [
        0.000, 0.447, 0.741,
        0.850, 0.325, 0.098,
        0.929, 0.694, 0.125,
        0.494, 0.184, 0.556,
        0.466, 0.674, 0.188,
        0.301, 0.745, 0.933,
        0.635, 0.078, 0.184,
        0.300, 0.300, 0.300,
        0.600, 0.600, 0.600,
        1.000, 0.000, 0.000,
        1.000, 0.500, 0.000,
        0.749, 0.749, 0.000,
        0.000, 1.000, 0.000,
        0.000, 0.000, 1.000,
        0.667, 0.000, 1.000,
        0.333, 0.333, 0.000,
        0.333, 0.667, 0.000,
        0.333, 1.000, 0.000,
        0.667, 0.333, 0.000,
        0.667, 0.667, 0.000,
        0.667, 1.000, 0.000,
        1.000, 0.333, 0.000,
        1.000, 0.667, 0.000,
        1.000, 1.000, 0.000,
        0.000, 0.333, 0.500,
        0.000, 0.667, 0.500,
        0.000, 1.000, 0.500,
        0.333, 0.000, 0.500,
        0.333, 0.333, 0.500,
        0.333, 0.667, 0.500,
        0.333, 1.000, 0.500,
        0.667, 0.000, 0.500,
        0.667, 0.333, 0.500,
        0.667, 0.667, 0.500,
        0.667, 1.000, 0.500,
        1.000, 0.000, 0.500,
        1.000, 0.333, 0.500,
        1.000, 0.667, 0.500,
        1.000, 1.000, 0.500,
        0.000, 0.333, 1.000,
        0.000, 0.667, 1.000,
        0.000, 1.000, 1.000,
        0.333, 0.000, 1.000,
        0.333, 0.333, 1.000,
        0.333, 0.667, 1.000,
        0.333, 1.000, 1.000,
        0.667, 0.000, 1.000,
        0.667, 0.333, 1.000,
        0.667, 0.667, 1.000,
        0.667, 1.000, 1.000,
        1.000, 0.000, 1.000,
        1.000, 0.333, 1.000,
        1.000, 0.667, 1.000,
        0.333, 0.000, 0.000,
        0.500, 0.000, 0.000,
        0.667, 0.000, 0.000,
        0.833, 0.000, 0.000,
        1.000, 0.000, 0.000,
        0.000, 0.167, 0.000,
        0.000, 0.333, 0.000,
        0.000, 0.500, 0.000,
        0.000, 0.667, 0.000,
        0.000, 0.833, 0.000,
        0.000, 1.000, 0.000,
        0.000, 0.000, 0.167,
        0.000, 0.000, 0.333,
        0.000, 0.000, 0.500,
        0.000, 0.000, 0.667,
        0.000, 0.000, 0.833,
        0.000, 0.000, 1.000,
        0.000, 0.000, 0.000,
        0.143, 0.143, 0.143,
        0.286, 0.286, 0.286,
        0.429, 0.429, 0.429,
        0.571, 0.571, 0.571,
        0.714, 0.714, 0.714,
        0.857, 0.857, 0.857,
        0.000, 0.447, 0.741,
        0.314, 0.717, 0.741,
        0.50, 0.5, 0
    ]
).astype(np.float32).reshape(-1, 3)

COCO_IDX= [
             1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 13, 14, 15, 16, 17,
            18, 19, 20, 21, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36,
            37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53,
            54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 67, 70, 72, 73,
            74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90
          ]

ANCHORS = { # stride: anchor
           8: [[1.25000, 1.62500], [2.00000, 3.75000], [4.12500, 2.87500]],
           16: [[1.87500, 3.81250], [3.87500, 2.81250], [3.68750, 7.43750]],
           32: [[ 3.62500,  2.81250], [ 4.87500,  6.18750], [11.65625, 10.18750]]
}

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def convert(o):
    if isinstance(o, np.int32): return float(o)
    raise TypeError

def get_image_id_in_path(image_path):
    stem = Path(image_path).stem
    # in val2014, name would be like COCO_val2014_000000xxxxxx.jpg
    # in val2017, name would be like 000000xxxxxx.jpg
    if (stem.rfind('_') == -1):
        id = int(stem)
    else:
        id = int(stem[stem.rfind('_') + 1:])
    return id

def get_img_id(file_name):
    ls = []
    myset = []
    annos = json.load(open(file_name, 'r'))
    for anno in annos:
        ls.append(anno['image_id'])
    myset = {}.fromkeys(ls).keys()
    return myset

def xyxy2xywh(bboxes):
    bboxes[:, 2] = bboxes[:, 2] - bboxes[:, 0]
    bboxes[:, 3] = bboxes[:, 3] - bboxes[:, 1]
    return bboxes

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
        cv2.rectangle(
            img,
            (x0, y0 + 1),
            (x0 + txt_size[0] + 1, y0 + int(1.5*txt_size[1])),
            txt_bk_color,
            -1
        )
        cv2.putText(img, text, (x0, y0 + txt_size[1]), font, 0.4, txt_color, thickness=1)

    return img

def nms(boxes, scores, nms_thr):
    """Single class NMS implemented in Numpy."""
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= nms_thr)[0]
        order = order[inds + 1]

    return keep

def multiclass_nms_class_aware(boxes, scores, nms_thr, score_thr):
    """Multiclass NMS implemented in Numpy. Class-aware version."""
    final_dets = []
    num_classes = scores.shape[1]
    for cls_ind in range(num_classes):
        cls_scores = scores[:, cls_ind]
        valid_score_mask = cls_scores > score_thr
        if valid_score_mask.sum() == 0:
            continue
        else:
            valid_scores = cls_scores[valid_score_mask]
            valid_boxes = boxes[valid_score_mask]
            keep = nms(valid_boxes, valid_scores, nms_thr)
            if len(keep) > 0:
                cls_inds = np.ones((len(keep), 1)) * cls_ind
                dets = np.concatenate(
                    [valid_boxes[keep], valid_scores[keep, None], cls_inds], 1
                )
                final_dets.append(dets)
    if len(final_dets) == 0:
        return None
    return np.concatenate(final_dets, 0)

def multiclass_nms_class_agnostic(boxes, scores, nms_thr, score_thr):
    """Multiclass NMS implemented in Numpy. Class-agnostic version."""
    cls_inds = scores.argmax(1)
    cls_scores = scores[np.arange(len(cls_inds)), cls_inds]

    valid_score_mask = cls_scores > score_thr
    if valid_score_mask.sum() == 0:
        return None
    valid_scores = cls_scores[valid_score_mask]
    valid_boxes = boxes[valid_score_mask]
    valid_cls_inds = cls_inds[valid_score_mask]
    keep = nms(valid_boxes, valid_scores, nms_thr)
    if keep:
        dets = np.concatenate(
            [valid_boxes[keep], valid_scores[keep, None], valid_cls_inds[keep, None]], 1
        )
    return dets

def multiclass_nms(boxes, scores, nms_thr, score_thr, class_agnostic=False):
    """Multiclass NMS implemented in Numpy"""
    if class_agnostic:
        nms_method = multiclass_nms_class_agnostic
    else:
        nms_method = multiclass_nms_class_aware
    return nms_method(boxes, scores, nms_thr, score_thr)

def preproc(img, input_size, swap=(2, 0, 1)):
    if len(img.shape) == 3:
        padded_img = np.ones((input_size[0], input_size[1], 3), dtype=np.uint8) * 114
    else:
        padded_img = np.ones(input_size, dtype=np.uint8) * 114

    r = min(input_size[0] / img.shape[0], input_size[1] / img.shape[1])
    resized_img = cv2.resize(
        img,
        (int(img.shape[1] * r), int(img.shape[0] * r)),
        interpolation=cv2.INTER_LINEAR  #INTER_AREA
    ).astype(np.uint8)
    padded_img[: int(img.shape[0] * r), : int(img.shape[1] * r)] = resized_img

    padded_img = padded_img.transpose(swap)[::-1]  # HWC to CHW, BGR to RGB
    padded_img = np.ascontiguousarray(padded_img, dtype=np.float32)
    return padded_img, r

def make_grid(nx, ny, stride, anchor):
    stride = np.array(stride)
    anchor = np.array(anchor)
    xv, yv = np.meshgrid(np.arange(nx),  np.arange(ny))
    grid = np.stack((xv, yv), -1)
    anchor_grid = (anchor * stride).reshape(1, len(anchor), 1, 1, 2)
    return grid, anchor_grid

def _sigmoid(x):
    return 1. / (1. + np.exp(-x))

def postproc(outputs, imsize, anchors = ANCHORS):
    z = []
    for out in  outputs:
        # bs, 3, 20, 20, 85
        if len(out.shape) == 5:
            bs, _, ny, nx,  _ = out.shape
        else:
            bs, _, ny, nx = out.shape
            out = out.reshape(bs, -1, 85, ny, nx)
            out = np.transpose(out, (0, 1,3,4,2))
        stride = imsize[0] / ny
        assert(stride == imsize[1] / nx)
        anchor = anchors[stride]
        grid, anchor_grid = make_grid(nx, ny, stride, anchor)
        y = _sigmoid(out)
        y[..., 0:2] = (y[..., 0:2] * 2 - 0.5 + grid) * stride  # xy
        y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * anchor_grid  # wh
        # batch = 1
        # z.append(y.view(bs, -1, 85))
        z.append(y.reshape(-1, 85))
    pred = np.concatenate(z, axis=0)
    boxes = pred[:, :4]
    scores = pred[:, 4:5] * pred[:, 5:]

    boxes_xyxy = np.ones_like(boxes)
    boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2]/2.
    boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3]/2.
    boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2]/2.
    boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3]/2.
    return scores, boxes_xyxy

def cal_coco_result(annotations_file, result_json_file):
    if not os.path.exists(result_json_file):
        assert ValueError("result file: {} not exist.".format(result_json_file))
    if not os.path.exists(annotations_file):
        assert ValueError("annotations_file file: {} not exist.".format(annotations_file))

    annType = ['segm', 'bbox', 'keypoints']
    annType = annType[1]
    cocoGt = COCO(annotations_file)
    imgIds = get_img_id(result_json_file)
    cocoDt = cocoGt.loadRes(result_json_file)
    imgIds = sorted(imgIds)
    imgIds = imgIds[0:5000]
    cocoEval = COCOeval(cocoGt, cocoDt, annType)
    cocoEval.params.imgIds = imgIds
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()

class coco_mAP(base_class):
    def init(self, args):
        self.args = args
        self.json_dict = []
        self.ratio_list = []

    def preproc(self, img_paths):
        img_list = []
        for path in img_paths.split(','):
            origin_img = cv2.imread(path)
            img, ratio = preproc(origin_img, self.args.net_input_dims)
            self.ratio_list.append(ratio)
            img = np.expand_dims(img, axis=0) / 255.  # 0 - 255 to 0.0 - 1.0
            img_list.append(img)
        x = np.concatenate(img_list, axis = 0)
        return x

    def update(self, idx, outputs, img_paths = None, labels = None, ratios = None):
        img_path_list = img_paths.split(',')
        # print(img_path_list, outputs[0].shape, outputs[1].shape, outputs[2].shape, self.args.net_input_dims)
        for i in range(outputs[0].shape[0]):
            batch1_output = []
            for output in outputs:
                batch1_output.append(np.expand_dims(output[i,...], axis=0))
            scores, boxes_xyxy = postproc(batch1_output, self.args.net_input_dims)
            dets = multiclass_nms(boxes_xyxy, scores, nms_thr=self.args.nms_threshold, score_thr=self.args.nms_score_thr)
            if dets is not None:
                final_boxes, final_scores, final_cls_inds = dets[:, :4], dets[:, 4], dets[:, 5]
                if ratios is not None:
                    final_boxes /= ratios[i]
                else:
                    final_boxes /= self.ratio_list[i]
                if idx < self.args.draw_image_count:
                    out_path = "vis_outout"
                    mkdir(out_path)
                    output_path = os.path.join(out_path, os.path.split(img_path_list[i])[-1])

                    origin_img = vis(origin_img, final_boxes, final_scores, final_cls_inds,
                                    conf=self.args.score_thr, class_names=COCO_CLASSES)
                    cv2.imwrite(output_path, origin_img)

                boxes_xywh = xyxy2xywh(final_boxes)
                # store js info
                try:
                    image_id = get_image_id_in_path(img_path_list[i])
                except ValueError:
                    image_id = None
                    print("Warning Make sure you are test custom image.")
                # convert to coco format
                for ind in range(final_boxes.shape[0]):
                    self.json_dict.append({
                        "image_id": image_id,
                        "category_id": COCO_IDX[final_cls_inds[ind].astype(np.int32)],
                        "bbox": list(boxes_xywh[ind]),
                        "score": float(final_scores[ind].astype(np.float32))
                    })

    def get_result(self):
        if os.path.exists('./result_json_file'):
            os.remove('./result_json_file')
        with open('./result_json_file', 'w') as file:
            json.dump(self.json_dict, file, default=convert)
        # eval coco
        cal_coco_result(self.args.coco_annotation, './result_json_file')

    def print_info(self):
        pass
