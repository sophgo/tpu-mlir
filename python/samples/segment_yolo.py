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
from tools.model_runner import mlir_inference, model_inference, onnx_inference, torch_inference
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


class PostProcess:

    def __init__(self, conf_thres=0.7, iou_thres=0.5, num_masks=32, fuse_postprocess=False):
        self.conf_threshold = conf_thres
        self.iou_threshold = iou_thres
        self.num_masks = num_masks
        self.nms = pseudo_torch_nms()
        self.fuse_postprocess = fuse_postprocess

    def __call__(self, outputs, im0_shape, ratio, txy):
        results = []
        if not self.fuse_postprocess:
            for out in outputs.values():
                assert (out.ndim == 3
                        or out.ndim == 4), "Wrong output, please provide a correct model"
                if (out.ndim == 3):
                    prediction = out
                if (out.ndim == 4):
                    proto = out

            for i in range(prediction.shape[0]):
                output = [prediction[i][np.newaxis, :], proto[i][np.newaxis, :]]
                results.append(
                    self.postprocess(output, im0_shape[i], ratio[i], txy[i][0], txy[i][1],
                                     self.conf_threshold, self.iou_threshold, self.num_masks))
            return results
        else:
            for out in outputs.values():
                if (out.ndim == 3):
                    masks_uncrop = out
                if (out.ndim == 2):
                    seg_out = out
            for i in range(len(txy)):
                masks, boxes = self.postprocess2(masks_uncrop, seg_out, im0_shape[i], ratio[i],
                                                 txy[i][0], txy[i][1])
                segments = self.masks2segments(masks)
                results.append([boxes, segments, masks])

            return results

    def postprocess(self,
                    preds,
                    im0_shape,
                    ratio,
                    pad_w,
                    pad_h,
                    conf_threshold,
                    iou_threshold,
                    nm=32):
        """
        Post-process the prediction.

        Args:
            preds (Numpy.ndarray): predictions come from ort.session.run().
            im0 (Numpy.ndarray): [h, w, c] original input image.
            ratio (tuple): width, height ratios in letterbox.
            pad_w (float): width padding in letterbox.
            pad_h (float): height padding in letterbox.
            conf_threshold (float): conf threshold.
            iou_threshold (float): iou threshold.
            nm (int): the number of masks.

        Returns:
            boxes (List): list of bounding boxes.
            segments (List): list of segments.
            masks (np.ndarray): [N, H, W], output masks.
        """
        x, protos = preds[0], preds[1]  # Two outputs: predictions and protos

        # Transpose the first output: (Batch_size, xywh_conf_cls_nm, Num_anchors) -> (Batch_size, Num_anchors, xywh_conf_cls_nm)
        x = np.einsum('bcn->bnc', x)

        # Predictions filtering by conf-threshold
        x = x[np.amax(x[..., 4:-nm], axis=-1) > conf_threshold]

        # Create a new matrix which merge these(box, score, cls, nm) into one
        # For more details about `numpy.c_()`: https://numpy.org/doc/1.26/reference/generated/numpy.c_.html
        x = np.c_[x[..., :4],
                  np.amax(x[..., 4:-nm], axis=-1),
                  np.argmax(x[..., 4:-nm], axis=-1), x[..., -nm:]]

        # NMS filtering
        if (x.shape[0]):
            x = x[self.nms.nms_boxes(x[:, :4], x[:, 4], iou_threshold)]

        ans1, ans2, ans3 = [], [], []
        post_batch_size = 1
        for i in range((int(x.shape[0] / post_batch_size) + 1)):
            X = x[i * post_batch_size:min((i + 1) * post_batch_size, x.shape[0])]
            X = self.get_mask_distrubute(X, im0_shape, ratio, pad_w, pad_h, protos)
            ans1.extend(X[0])
            ans2.extend(X[1])
            ans3.extend(X[2])
        return ans1, ans2, ans3

    def postprocess2(self, masks_uncrop, seg_out, im0_shape, ratio, pad_w, pad_h):

        seg_out = seg_out[:-1, ...]
        masks_uncrop = masks_uncrop[:-1, ...]
        masks_uncrop = masks_uncrop.transpose(1, 2, 0)
        masks_uncrop = self.scale_mask(masks_uncrop, im0_shape[:2])
        masks_uncrop = masks_uncrop.transpose(2, 0, 1)
        boxes = seg_out[:, :4]

        boxes[..., :4] -= [pad_w, pad_h, pad_w, pad_h]
        boxes[..., :4] /= min(ratio)

        boxes[..., [0, 2]] = boxes[:, [0, 2]].clip(0, im0_shape[:2][1])
        boxes[..., [1, 3]] = boxes[:, [1, 3]].clip(0, im0_shape[:2][0])
        masks = self.crop_mask(masks_uncrop, boxes)
        masks = np.greater(masks, 0.5)
        return masks, seg_out

    def get_mask_distrubute(self, x, im0_shape, ratio, pad_w, pad_h, protos):
        if len(x) > 0:
            x[..., [0, 1]] -= x[..., [2, 3]] / 2
            x[..., [2, 3]] += x[..., [0, 1]]

            x[..., :4] -= [pad_w, pad_h, pad_w, pad_h]
            x[..., :4] /= min(ratio)

            x[..., [0, 2]] = x[:, [0, 2]].clip(0, im0_shape[1])
            x[..., [1, 3]] = x[:, [1, 3]].clip(0, im0_shape[0])
            masks = self.process_mask(protos[0], x[:, 6:], x[:, :4], im0_shape)

            # Masks -> Segments(contours)
            segments = self.masks2segments(masks)
            return x[..., :6], segments, masks  # boxes, segments, masks
        else:
            return [], [], []

    @staticmethod
    def masks2segments(masks):

        segments = []
        for x in masks.astype('uint8'):
            contours, _ = cv2.findContours(x, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            if (contours):
                contours = np.array(contours[np.array([len(x) for x in contours]).argmax()])
                coco_segmentation = [contours.flatten().astype('float32')]
                segments.append(coco_segmentation)
            else:
                segments.append([])

        return segments

    @staticmethod
    def crop_mask(masks, boxes):
        n, h, w = masks.shape
        x1, y1, x2, y2 = np.split(boxes[:, :, None], 4, 1)
        r = np.arange(w, dtype=x1.dtype)[None, None, :]
        c = np.arange(h, dtype=x1.dtype)[None, :, None]
        return masks * ((r >= x1) * (r < x2) * (c >= y1) * (c < y2))

    def process_mask(self, protos, masks_in, bboxes, im0_shape):
        """
        Takes the output of the mask head, and applies the mask to the bounding boxes. This produces masks of higher quality
        but is slower. (Borrowed from https://github.com/ultralytics/ultralytics/blob/465df3024f44fa97d4fad9986530d5a13cdabdca/ultralytics/utils/ops.py#L618)

        Args:
            protos (numpy.ndarray): [mask_dim, mask_h, mask_w].
            masks_in (numpy.ndarray): [n, mask_dim], n is number of masks after nms.
            bboxes (numpy.ndarray): bboxes re-scaled to original image shape.
            im0_shape (tuple): the size of the input image (h,w,c).

        Returns:
            (numpy.ndarray): The upsampled masks.
        """
        c, mh, mw = protos.shape
        masks = np.matmul(masks_in, protos.reshape((c, -1))).reshape(
            (-1, mh, mw)).transpose(1, 2, 0)  # HWN
        masks = np.ascontiguousarray(masks)
        masks = self.scale_mask(
            masks, im0_shape)  # re-scale mask from P3 shape to original input image shape
        masks = np.einsum('HWN -> NHW', masks)  # HWN -> NHW
        masks = self.crop_mask(masks, bboxes)
        return np.greater(masks, 0.5)

    @staticmethod
    def scale_mask(masks, im0_shape, ratio_pad=None):
        """
        Takes a mask, and resizes it to the original image size. (Borrowed from
        https://github.com/ultralytics/ultralytics/blob/465df3024f44fa97d4fad9986530d5a13cdabdca/ultralytics/utils/ops.py#L305)

        Args:
            masks (np.ndarray): resized and padded masks/images, [h, w, num]/[h, w, 3].
            im0_shape (tuple): the original image shape.
            ratio_pad (tuple): the ratio of the padding to the original image.

        Returns:
            masks (np.ndarray): The masks that are being returned.
        """
        im1_shape = masks.shape[:2]

        if ratio_pad is None:  # calculate from im0_shape
            gain = min(im1_shape[0] / im0_shape[0],
                       im1_shape[1] / im0_shape[1])  # gain  = old / new
            pad = (im1_shape[1] - im0_shape[1] * gain) / 2, (im1_shape[0] -
                                                             im0_shape[0] * gain) / 2  # wh padding
        else:
            pad = ratio_pad[1]

        # Calculate tlbr of mask
        top, left = int(round(pad[1] - 0.1)), int(round(pad[0] - 0.1))  # y, x
        bottom, right = int(round(im1_shape[0] - pad[1] + 0.1)), int(
            round(im1_shape[1] - pad[0] + 0.1))
        if len(masks.shape) < 2:
            raise ValueError(f'"len of masks shape" should be 2 or 3, but got {len(masks.shape)}')
        masks = masks[top:bottom, left:right]
        masks = cv2.resize(masks, (im0_shape[1], im0_shape[0]))  #,
        #interpolation=cv2.INTER_CUBIC)  # INTER_CUBIC would be better
        if len(masks.shape) == 2:
            masks = masks[:, :, None]
        return masks


class pseudo_torch_nms:

    def nms_boxes(self, boxes, scores, iou_thres):
        x = boxes[:, 0]
        y = boxes[:, 1]
        w = boxes[:, 2]
        h = boxes[:, 3]

        areas = w * h
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)

            xx1 = np.maximum(x[i], x[order[1:]])
            yy1 = np.maximum(y[i], y[order[1:]])
            xx2 = np.minimum(x[i] + w[i], x[order[1:]] + w[order[1:]])
            yy2 = np.minimum(y[i] + h[i], y[order[1:]] + h[order[1:]])

            w1 = np.maximum(0.0, xx2 - xx1 + 0.00001)
            h1 = np.maximum(0.0, yy2 - yy1 + 0.00001)
            inter = w1 * h1

            ovr = inter / (areas[i] + areas[order[1:]] - inter)
            inds = np.where(ovr <= iou_thres)[0]
            order = order[inds + 1]
        keep = np.array(keep)
        return keep


def preprocess(ori_img, input_shape, fuse_preprocess=False):

    letterbox_img, ratio, (tx1, ty1) = letterbox(ori_img,
                                                 new_shape=input_shape,
                                                 color=(114, 114, 114),
                                                 auto=False,
                                                 scaleFill=False,
                                                 scaleup=True,
                                                 stride=32)

    img = letterbox_img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
    img = img.astype(np.float32)
    if not fuse_preprocess:
        img = np.ascontiguousarray(img / 255.0)
    return img, ratio, (tx1, ty1)


def letterbox(im,
              new_shape=(640, 640),
              color=(114, 114, 114),
              auto=False,
              scaleFill=False,
              scaleup=True,
              stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT,
                            value=color)  # add border

    return im, ratio, (dw, dh)


def hsv2bgr(h, s, v):
    h_i = int(h * 6)
    f = h * 6 - h_i
    p = v * (1 - s)
    q = v * (1 - f * s)
    t = v * (1 - (1 - f) * s)

    r, g, b = 0, 0, 0

    if h_i == 0:
        r, g, b = v, t, p
    elif h_i == 1:
        r, g, b = q, v, p
    elif h_i == 2:
        r, g, b = p, v, t
    elif h_i == 3:
        r, g, b = p, q, v
    elif h_i == 4:
        r, g, b = t, p, v
    elif h_i == 5:
        r, g, b = v, p, q

    return int(b * 255), int(g * 255), int(r * 255)


def random_color(id):
    h_plane = (((id << 2) ^ 0x937151) % 100) / 100.0
    s_plane = (((id << 3) ^ 0x315793) % 100) / 100.0
    return hsv2bgr(h_plane, s_plane, 1)


def draw_and_visualize(filename, im, bboxes, segments, vis=False, save=True, using_COCO_name=True):

    # Draw rectangles and polygons
    im_canvas = im.copy()
    for (*box, confidence, label), segment in zip(bboxes, segments):

        if confidence < 0.25: continue
        color = random_color(int(label))
        #draw contour and fill mask
        if (len(segment)):
            for seg in segment:
                # cv2.polylines(im, np.int32([np.int32([seg]).reshape(-1,1,2)]), True, color, 2)  # white borderline
                cv2.fillPoly(im_canvas, np.int32([np.int32([seg]).reshape(-1, 1, 2)]), color)

        # draw bbox rectangle
        left, top, right, bottom = int(box[0]), int(box[1]), int(box[2]), int(box[3])

        cv2.rectangle(im, (left, top), (right, bottom),
                      color=color,
                      thickness=2,
                      lineType=cv2.LINE_AA)
        if using_COCO_name:
            caption = f"{COCO_CLASSES[int(label)]} {confidence:.3f}"
        else:
            caption = f"class:{int(label)} {confidence:.3f}"
        w, h = cv2.getTextSize(caption, 0, 1, 2)[0]
        cv2.rectangle(im, (left - 3, top - 33), (left + w + 10, top), color, -1)
        cv2.putText(im, caption, (left, top - 5), 0, 1, (0, 0, 0), 2, 16)

    # Mix image
    im = cv2.addWeighted(im_canvas, 0.3, im, 0.7, 0)

    if save:
        cv2.imwrite(filename, im)
        print(f"output been saved as {filename}")
    return im


def parse_args():
    # yapf: disable
    parser = argparse.ArgumentParser(description='Inference demo of Yolo_seg v8/v11 network.')
    parser.add_argument("--model", type=str, required=True, help="Model definition file")
    parser.add_argument("--net_input_dims", type=str, default="640,640", help="(h,w) of net input")
    parser.add_argument("--input", type=str, required=True, help="Input image for testing")
    parser.add_argument("--output", type=str, required=True, help="Output image after detection")
    parser.add_argument("--conf_thres", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--iou_thres", type=float, default=0.7, help="NMS IOU threshold")
    parser.add_argument("--score_thres", type=float, default=0.5, help="Score of the result")
    parser.add_argument("--fuse_preprocess", action='store_true', help="if the model fused preprocess")
    parser.add_argument("--fuse_postprocess", action='store_true', help="if the model fused postprocess")
    # yapf: enable
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    input_shape = tuple(map(int, args.net_input_dims.split(',')))
    origin_img = cv2.imread(args.input)
    img, ratio, (tx1, ty1) = preprocess(origin_img, input_shape, args.fuse_preprocess)  #new preproc
    ratio_list = []
    txy_list = []
    ori_size_list = []
    ratio_list.append(ratio)
    txy_list.append([tx1, ty1])
    ori_h, ori_w = origin_img.shape[:2]
    ori_size_list.append((ori_h, ori_w))
    img = np.expand_dims(img, axis=0)
    data = {"data": img}  # input name from model
    output = dict()
    if args.model.endswith('.onnx'):
        output = onnx_inference(data, args.model, False)
    elif args.model.endswith('.pt') or args.model.endswith('.pth'):
        output = torch_inference(data, args.model, False)
    elif args.model.endswith('.mlir'):
        output = mlir_inference(data, args.model, False)
    elif args.model.endswith(".bmodel"):
        output = model_inference(data, args.model)
    elif args.model.endswith(".cvimodel"):
        output = model_inference(data, args.model, False)
    else:
        raise RuntimeError("not support modle file:{}".format(args.model))

    assert len(
        output.values()) == 2, "The number of outputs should be 2, please provide a correct model"
    postprocess = PostProcess(conf_thres=args.conf_thres,
                              iou_thres=args.iou_thres,
                              fuse_postprocess=args.fuse_postprocess)

    results = postprocess(output, ori_size_list, ratio_list, txy_list)
    boxes, segments, masks = results[0]

    draw_and_visualize(args.output, origin_img, boxes, segments, vis=False, save=True)


if __name__ == "__main__":
    main()
