# SPDX-License-Identifier: MIT

import cv2
import onnxruntime as ort
import argparse
import numpy as np
from tools.model_runner import mlir_inference, model_inference, onnx_inference


def area_of(left_top, right_bottom):
    """
    Compute the areas of rectangles given two corners.
    Args:
        left_top (N, 2): left top corner.
        right_bottom (N, 2): right bottom corner.
    Returns:
        area (N): return the area.
    """
    hw = np.clip(right_bottom - left_top, 0.0, None)
    return hw[..., 0] * hw[..., 1]


def iou_of(boxes0, boxes1, eps=1e-5):
    """
    Return intersection-over-union (Jaccard index) of boxes.
    Args:
        boxes0 (N, 4): ground truth boxes.
        boxes1 (N or 1, 4): predicted boxes.
        eps: a small number to avoid 0 as denominator.
    Returns:
        iou (N): IoU values.
    """
    overlap_left_top = np.maximum(boxes0[..., :2], boxes1[..., :2])
    overlap_right_bottom = np.minimum(boxes0[..., 2:], boxes1[..., 2:])

    overlap_area = area_of(overlap_left_top, overlap_right_bottom)
    area0 = area_of(boxes0[..., :2], boxes0[..., 2:])
    area1 = area_of(boxes1[..., :2], boxes1[..., 2:])
    return overlap_area / (area0 + area1 - overlap_area + eps)


def hard_nms(box_scores, iou_threshold, top_k=-1, candidate_size=200):
    """
    Perform hard non-maximum-supression to filter out boxes with iou greater
    than threshold
    Args:
        box_scores (N, 5): boxes in corner-form and probabilities.
        iou_threshold: intersection over union threshold.
        top_k: keep top_k results. If k <= 0, keep all the results.
        candidate_size: only consider the candidates with the highest scores.
    Returns:
        picked: a list of indexes of the kept boxes
    """
    scores = box_scores[:, -1]
    boxes = box_scores[:, :-1]
    picked = []
    indexes = np.argsort(scores)
    indexes = indexes[-candidate_size:]
    while len(indexes) > 0:
        current = indexes[-1]
        picked.append(current)
        if 0 < top_k == len(picked) or len(indexes) == 1:
            break
        current_box = boxes[current, :]
        indexes = indexes[:-1]
        rest_boxes = boxes[indexes, :]
        iou = iou_of(
            rest_boxes,
            np.expand_dims(current_box, axis=0),
        )
        indexes = indexes[iou <= iou_threshold]

    return box_scores[picked, :]


def predict(width, height, confidences, boxes, prob_threshold, iou_threshold=0.5, top_k=-1):
    """
    Select boxes that contain human faces
    Args:
        width: original image width
        height: original image height
        confidences (N, 2): confidence array
        boxes (N, 4): boxes array in corner-form
        iou_threshold: intersection over union threshold.
        top_k: keep top_k results. If k <= 0, keep all the results.
    Returns:
        boxes (k, 4): an array of boxes kept
        labels (k): an array of labels for each boxes kept
        probs (k): an array of probabilities for each boxes being in corresponding labels
    """
    boxes = boxes[0]
    confidences = confidences[0]
    #print(boxes)
    #print(confidences)

    picked_box_probs = []
    picked_labels = []
    for class_index in range(1, confidences.shape[1]):
        #print(confidences.shape[1])
        probs = confidences[:, class_index]
        #print(probs)
        mask = probs > prob_threshold
        probs = probs[mask]

        if probs.shape[0] == 0:
            continue
        subset_boxes = boxes[mask, :]
        #print(subset_boxes)
        box_probs = np.concatenate([subset_boxes, probs.reshape(-1, 1)], axis=1)
        box_probs = hard_nms(
            box_probs,
            iou_threshold=iou_threshold,
            top_k=top_k,
        )
        picked_box_probs.append(box_probs)
        picked_labels.extend([class_index] * box_probs.shape[0])
    if not picked_box_probs:
        return np.array([]), np.array([]), np.array([])
    picked_box_probs = np.concatenate(picked_box_probs)
    picked_box_probs[:, 0] *= width
    picked_box_probs[:, 1] *= height
    picked_box_probs[:, 2] *= width
    picked_box_probs[:, 3] *= height
    return picked_box_probs[:, :4].astype(np.int32), np.array(picked_labels), picked_box_probs[:, 4]


def preprocess(orig_image, input_shape):
    image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, input_shape)
    image_mean = np.array([127, 127, 127])
    image = (image - image_mean) / 128
    image = np.transpose(image, [2, 0, 1])
    image = np.expand_dims(image, axis=0)
    image = image.astype(np.float32)
    return image


# scale current rectangle to box
def scale(box):
    width = box[2] - box[0]
    height = box[3] - box[1]
    maximum = max(width, height)
    dx = int((maximum - width) / 2)
    dy = int((maximum - height) / 2)

    bboxes = [box[0] - dx, box[1] - dy, box[2] + dx, box[3] + dy]
    return bboxes


# crop image
def cropImage(image, box):
    num = image[box[1]:box[3], box[0]:box[2]]
    return num


def parse_args():
    parser = argparse.ArgumentParser(description='Inference inception_v3 network.')
    parser.add_argument("--model", type=str, required=True, help="Model definition file")
    parser.add_argument("--model_data", type=str, help="Caffemodel data, only for caffe model")
    parser.add_argument("--net_input_dims", type=str, default="640,480", help="(h,w) of net input")
    parser.add_argument("--input", type=str, required=True, help="Input image for testing")
    parser.add_argument("--output",
                        type=str,
                        required=True,
                        help="Output image after classification")
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    input_shape = tuple(map(int, args.net_input_dims.split(',')))

    origin_image = cv2.imread(args.input)
    color = (255, 128, 0)
    image_data = preprocess(origin_image, input_shape)
    data = {"input": image_data}  # input name from model

    output = dict()
    if args.model.endswith('.onnx'):
        output = onnx_inference(data, args.model, False)
        confidences = output['scores']
        boxes = output['boxes']

        boxes, label, probs = predict(origin_image.shape[1], origin_image.shape[0], confidences,
                                      boxes, 0.7)

    else:
        if args.model.endswith('.mlir'):
            output = mlir_inference(data, args.model, False)
        elif args.model.endswith(".bmodel"):
            output = model_inference(data, args.model)
        else:
            raise RuntimeError("not support modle file:{}".format(args.model))

        confidences = output['scores_Softmax']
        boxes = output['boxes_Concat']

        boxes, label, probs = predict(origin_image.shape[1], origin_image.shape[0], confidences,
                                      boxes, 0.7)
        # boxes, categories, confidences = postprocess(output, input_shape, origin_image.size,
        #                                              args.score_thres, args.iou_thres,
        #                                              args.conf_thres)

    for i in range(boxes.shape[0]):
        box = scale(boxes[i, :])
        cv2.rectangle(origin_image, (box[0], box[1]), (box[2], box[3]), color, 4)
        # cv2.imshow('', origin_image)
        cv2.imwrite(f'{i}.jpg', origin_image)


# ------------------------------------------------------------------------------------------------------------------------------------------------
# Main void

if __name__ == '__main__':
    main()
