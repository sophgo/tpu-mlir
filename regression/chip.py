#!/usr/bin/env python3
# Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
#
# TPU-MLIR is licensed under the 2-Clause BSD License except for the
# third-party components.
#
# ==============================================================================

# yapf: disable
######################################
# Chip Support
######################################
Y, N = True, False

chip_support = {
    # chip_name: (F32, F16, BF16, INT8_sym, INT8_asym, INT4_sym, dynamic, model_type)
    "bm1684":    (Y,   N,   N,    Y,        N,         N,        N,       "bmodel"),
    "bm1684x":   (Y,   Y,   Y,    Y,        Y,         N,        N,       "bmodel"),
    "bm1686":    (Y,   Y,   Y,    Y,        Y,         Y,        N,       "bmodel"),
    "cv180x":    (N,   N,   Y,    Y,        N,         N,        N,       "cvimodel"),
    "cv181x":    (N,   N,   Y,    Y,        N,         N,        N,       "cvimodel"),
    "cv182x":    (N,   N,   Y,    Y,        N,         N,        N,       "cvimodel"),
    "cv183x":    (N,   N,   Y,    Y,        N,         N,        N,       "cvimodel"),
}

'''
    basic_model_list is for each commit test
    full_model_list is for daily test

    Note:
        1. suffix: _pt = pytorch, _cf = caffe, _tf = tflite, default is onnx
        2. order of chips in model list should keep the same as in chip_support
'''

######################################
# Model Support
######################################
basic_model_list = {
    # model_name:                (bm1684, bm1684x, bm1686, cv180x, cv181x, cv182x, cv183x)
    "mobilenet_v2_cf":            (N,      Y,       N,      Y,      N,      Y,      Y),
    "resnet50_v2":                (Y,      N,       N,      N,      N,      Y,      Y),
    "yolov5s":                    (N,      Y,       Y,      N,      N,      Y,      Y),
    "yolov5s_pt":                 (N,      Y,       N,      N,      N,      N,      N),
    "yolov5s_tf":                 (N,      Y,       N,      N,      N,      N,      N),
}


full_model_list = {
    # model_name:                (bm1684, bm1684x, bm1686, cv180x, cv181x, cv182x, cv183x)
    ######## onnx ###############
    "bert-tiny_from_pt":          (N,      Y,       Y,      N,      N,      Y,      Y),
    "densenet121-12":             (N,      Y,       Y,      N,      N,      Y,      Y),
    "efficientnet":               (N,      Y,       Y,      N,      N,      Y,      Y),
    "inception_v3":               (N,      Y,       Y,      N,      N,      Y,      Y),
    "mnist-12":                   (N,      Y,       Y,      N,      N,      Y,      Y),
    "mobilenet_v2":               (N,      Y,       Y,      N,      N,      Y,      Y),
    "resnet18_v1":                (N,      Y,       Y,      N,      N,      Y,      Y),
    "resnet18_v2":                (N,      Y,       Y,      N,      N,      Y,      Y),
    "resnet50_v1":                (N,      Y,       Y,      N,      N,      Y,      Y),
    "resnet50_v2":                (Y,      Y,       Y,      N,      N,      Y,      Y),
    "retinaface":                 (N,      Y,       Y,      N,      N,      Y,      Y),
    "se-resnet50":                (N,      Y,       Y,      N,      N,      Y,      Y),
    "shufflenet_v2":              (N,      Y,       Y,      N,      N,      Y,      Y),
    "squeezenet1.0":              (N,      Y,       Y,      N,      N,      Y,      Y),
    "ssd-12":                     (N,      Y,       Y,      N,      N,      N,      N),
    "ultraface_640":              (N,      Y,       Y,      N,      N,      N,      N),
    "vgg16":                      (N,      Y,       Y,      N,      N,      Y,      Y),
    "yolov5s":                    (N,      Y,       Y,      N,      N,      Y,      Y),
    "yolov3_tiny":                (N,      Y,       Y,      N,      N,      Y,      Y),
    "yolox_s":                    (N,      Y,       Y,      N,      N,      Y,      Y),
    "yolov5s_with_trans":         (N,      Y,       Y,      N,      N,      Y,      Y),
    "alphapose_res50":            (N,      N,       N,      N,      N,      Y,      Y),
    ######## Pytorch #######      ######
    "bert_pt":                    (N,      Y,       Y,      N,      N,      N,      N),
    "bert_base_pt":               (N,      N,       N,      N,      N,      N,      N),
    "bert_large_pt":              (N,      N,       N,      N,      N,      N,      N),
    "resnet50_pt":                (N,      Y,       Y,      N,      N,      Y,      Y),
    "sd_encoder_pt":              (N,      Y,       Y,      N,      N,      N,      N),
    "sd_decoder_pt":              (N,      Y,       Y,      N,      N,      N,      N),
    "yolov5s_pt":                 (N,      Y,       Y,      N,      N,      Y,      Y),
    ######## caffe #########      #####
    "mobilenet_v2_cf":            (N,      Y,       Y,      N,      N,      Y,      Y),
    "resnet18_cf":                (N,      Y,       Y,      N,      N,      Y,      Y),
    "segnet_cf":                  (N,      Y,       Y,      N,      N,      Y,      Y),
    "mobilenet_v2_cf_bs4":        (N,      N,       N,      N,      N,      Y,      Y),
    "retinaface_mnet_with_det":   (N,      N,       N,      Y,      N,      Y,      Y),
    "arcface_res50":              (N,      Y,       Y,      N,      N,      Y,      Y),
    "yolov3_416_with_det":        (N,      N,       N,      N,      N,      Y,      Y),
    # ----- cvs20-test -----      -------
    "feature_extract_cf":         (N,      Y,       Y,      N,      N,      Y,      Y),
    "squeezenet_v1.1_cf":         (N,      Y,       Y,      N,      N,      Y,      Y),
    ######## tflite ########      ########
    "inception_v4_tf":            (N,      Y,       Y,      N,      N,      N,      N),
    "mobilenet_v2_tf":            (N,      Y,       Y,      N,      N,      N,      N),
    "resnet50_tf":                (N,      Y,       Y,      N,      N,      N,      N),
    "ssd_mobilenet_v1_tf":        (N,      Y,       Y,      N,      N,      N,      N),
    "yolov5s_tf":                 (N,      Y,       Y,      N,      N,      N,      N),
    "mobilebert_tf":              (N,      Y,       Y,      N,      N,      N,      N),
    ######## PaddlePaddle ##      ##############
    "pp_humanseg":                (N,      Y,       Y,      N,      N,      N,      N),
    "pp_liteseg":                 (N,      Y,       Y,      N,      N,      N,      N),
    "pp_picodet":                 (N,      Y,       Y,      N,      N,      Y,      Y),
    "pp_yoloe":                   (N,      Y,       Y,      N,      N,      Y,      Y),
    "pp_yolox":                   (N,      Y,       Y,      N,      N,      Y,      Y),
    "pp_yolov3":                  (N,      Y,       Y,      N,      N,      Y,      Y),
    "pp_ocr_det":                 (N,      Y,       Y,      N,      N,      Y,      Y),
    "pp_ocr_cls":                 (N,      Y,       Y,      N,      N,      Y,      Y),
    "pp_ocr_rec":                 (N,      Y,       Y,      N,      N,      N,      N),
    "pp_hgnet":                   (N,      Y,       Y,      N,      N,      Y,      Y),
}
