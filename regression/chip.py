#!/usr/bin/env python3
# Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
#
# TPU-MLIR is licensed under the 2-Clause BSD License except for the
# third-party components.
#
# ==============================================================================

# yapf: disable
######################################
# chip support
######################################
Y, N = True, False

chip_support = {
    #########################################
    # Supported Quant Modes and Model Type
    #########################################
    # chip_name: (F32, F16, BF16, INT8_sym, INT8_asym, INT4_sym, dynamic, model_type)
    "bm1684":    (Y,   N,   N,    Y,        N,         N,        N,       "bmodel"),
    "bm1684x":   (Y,   Y,   Y,    Y,        Y,         N,        N,       "bmodel"),
    "bm1686":    (Y,   Y,   Y,    Y,        N,         N,        N,       "bmodel"),
    "cv180x":    (N,   N,   Y,    Y,        N,         N,        N,       "cvimodel"),
    "cv181x":    (N,   N,   Y,    Y,        N,         N,        N,       "cvimodel"),
    "cv182x":    (N,   N,   Y,    Y,        N,         N,        N,       "cvimodel"),
    "cv183x":    (N,   N,   Y,    Y,        N,         N,        N,       "cvimodel"),
}

# basic is for each commit test
# all is for daily test


######################################
# model support
######################################

bm1684_model_list = {
    # _cf = caffe, _tf = tflite, default is onnx
    # model_name:      (basic, all)
    ######## onnx ###############
    "mobilenet_v2":        (N, N),
    "resnet18_v1":         (N, N),
    "resnet18_v2":         (N, N),
    "resnet50_v1":         (N, N),
    "resnet50_v2":         (Y, Y),
    "vgg16":               (N, N),
    "yolov5s":             (N, N),
    ######## caffe ##############
    "mobilenet_v2_cf":     (N, N),
    "resnet18_cf":         (N, N),
    "segnet_cf":           (N, N),
}

bm1684x_model_list = {
    # _pt = pytorch, _cf = caffe, _tf = tflite, default is onnx
    # model_name:      (basic, all)
    ######## Pytorch #############
    "bert_pt":             (N, Y),
    "resnet50_pt":         (N, Y),
    "sd_encoder_pt":       (N, Y),
    "sd_decoder_pt":       (N, Y),
    "yolov5s_pt":          (Y, Y),
    ######## onnx ################
    "bert-tiny_from_pt":   (N, Y),
    "densenet121-12":      (N, Y),
    "efficientnet":        (N, Y),
    "inception_v3":        (N, Y),
    "mnist-12":            (N, Y),
    "mobilenet_v2":        (N, Y),
    "resnet18_v1":         (N, Y),
    "resnet18_v2":         (N, Y),
    "resnet50_v1":         (N, Y),
    "resnet50_v2":         (N, Y),
    "retinaface":          (N, Y),
    "se-resnet50":         (N, Y),
    "shufflenet_v2":       (N, Y),
    "squeezenet1.0":       (N, Y),
    "ssd-12":              (N, Y),
    "ultraface_640":       (N, Y),
    "vgg16":               (N, Y),
    "yolov5s":             (Y, Y),
    "yolov3_tiny":         (N, Y),
    ######## caffe ################
    "mobilenet_v2_cf":     (Y, Y),
    "resnet18_cf":         (N, Y),
    "segnet_cf":           (N, Y),
    "feature_extract_cf":  (N, Y),
    "squeezenet_v1.1_cf":  (N, Y),
    ######## tflite ################
    "inception_v4_tf":     (N, Y),
    "mobilenet_v2_tf":     (N, Y),
    "resnet50_tf":         (N, Y),
    "ssd_mobilenet_v1_tf": (N, Y),
    "yolov5s_tf":          (Y, Y),
    "mobilebert_tf":       (N, Y),
    ######## PaddlePaddle ##########
    "pp_humanseg":         (N, Y),
    "pp_liteseg":          (N, Y),
    "pp_picodet":          (N, Y),
    "pp_yoloe":            (N, Y),
    "pp_yolox":            (N, Y),
    "pp_yolov3":           (N, Y),
    "pp_ocr_det":          (N, Y),
    "pp_ocr_cls":          (N, Y),
    "pp_ocr_rec":          (N, Y),
    "pp_hgnet":            (N, Y),
}

bm1686_model_list = {
    # _cf = caffe, _tf = tflite, default is onnx
    # model_name:      (basic, all)
    ######## onnx ################
    "bert-tiny_from_pt":   (N, N),
    "densenet121-12":      (N, Y),
    "efficientnet":        (N, Y),
    "inception_v3":        (N, Y),
    "mnist-12":            (N, Y),
    "mobilenet_v2":        (N, Y),
    "resnet18_v1":         (N, Y),
    "resnet18_v2":         (N, Y),
    "resnet50_v1":         (N, Y),
    "resnet50_v2":         (N, Y),
    "retinaface":          (N, Y),
    "se-resnet50":         (N, Y),
    "shufflenet_v2":       (N, Y),
    "squeezenet1.0":       (N, Y),
    "ssd-12":              (N, Y),
    "ultraface_640":       (N, N),
    "vgg16":               (N, Y),
    "yolov5s":             (Y, Y),
    "yolov3_tiny":         (N, Y),
    ######## caffe ################
    "mobilenet_v2_cf":     (N, Y),
    "resnet18_cf":         (N, Y),
    "segnet_cf":           (N, N),
    # ----- cvs20-test ------------
    "feature_extract_cf":  (N, Y),
    "squeezenet_v1.1_cf":  (N, Y),
    ######## tflite ################
    "inception_v4_tf":     (N, N),
    "mobilenet_v2_tf":     (N, N),
    "resnet50_tf":         (N, N),
    "ssd_mobilenet_v1_tf": (N, N),
    "yolov5s_tf":          (N, N),
    "mobilebert_tf":       (N, N),
    ######## PaddlePaddle ################
    "pp_humanseg":         (N, N),
    "pp_liteseg":          (N, N),
    "pp_picodet":          (N, Y),
    "pp_yoloe":            (N, N),
    "pp_yolox":            (N, Y),
    "pp_yolov3":           (N, N),
    "pp_ocr_det":          (N, N),
    "pp_ocr_cls":          (N, N),
    "pp_ocr_rec":          (N, Y),
}

cv180x_model_list = {
    # _pt = pytorch, _cf = caffe, _tf = tflite, default is onnx
    # model_name:      (basic, all)
    "mobilenet_v2_cvi":    (Y, Y),
    "retinaface_mnet_cvi": (Y, Y),
  }

cv181x_model_list = {
    # _pt = pytorch, _cf = caffe, _tf = tflite, default is onnx
    # model_name:      (basic, all)
    "mobilenet_v2_cvi":    (Y, Y),
    "mobilenet_v2_cvi_bs4":(Y, Y),
    "retinaface_mnet_cvi": (Y, Y),
    "yolox_s_cvi":         (Y, Y),
    "yolov5s_cvi":         (Y, Y),
    "alphapose_res50_cvi": (Y, Y),
    "arcface_res50_cvi":   (Y, Y),
}

cv182x_model_list = {
    # _pt = pytorch, _cf = caffe, _tf = tflite, default is onnx
    # model_name:      (basic, all)
    "mobilenet_v2_cvi":    (Y, Y),
    "mobilenet_v2_cvi_bs4":(Y, Y),
    "retinaface_mnet_cvi": (Y, Y),
    "yolov3_416_cvi":      (Y, Y),
    "yolox_s_cvi":         (Y, Y),
    "yolov5s_cvi":         (Y, Y),
    "alphapose_res50_cvi": (Y, Y),
    "arcface_res50_cvi":   (Y, Y),
}

cv183x_model_list = {
    # _pt = pytorch, _cf = caffe, _tf = tflite, default is onnx
    # model_name:      (basic, all)
    ######## onnx ################
    "densenet121-12":      (N, Y),
    "efficientnet":        (N, Y),
    "inception_v3":        (N, Y),
    "retinaface":          (N, N),
    "mnist-12":            (N, Y),
    "mobilenet_v2":        (N, Y),
    "resnet18_v1":         (N, Y),
    "resnet18_v2":         (N, Y),
    "resnet50_v1":         (N, Y),
    "resnet50_v2":         (N, Y),
    "se-resnet50":         (N, Y),
    "shufflenet_v2":       (N, Y),
    "squeezenet1.0":       (N, Y),
    "vgg16":               (N, Y),
    ######## caffe ################
    "resnet18_cf":         (N, Y),
    # object detection
    "ssd-12":              (N, N),
    "yolov5s":             (Y, Y),
    "yolov3_tiny":         (N, Y),
    # cvimodel samples
    "mobilenet_v2_cvi":    (Y, Y),
    "mobilenet_v2_cvi_bs4":(Y, Y),
    "retinaface_mnet_cvi": (Y, Y),
    "yolov3_416_cvi":      (Y, Y),
    "yolox_s_cvi":         (Y, Y),
    "yolov5s_cvi":         (Y, Y),
    "alphapose_res50_cvi": (Y, Y),
    "arcface_res50_cvi":   (Y, Y),
}
