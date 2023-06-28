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
    "bm1684x":   (Y,   Y,   Y,    Y,        N,         N,        N,       "bmodel"),
    "bm1686":    (Y,   Y,   Y,    Y,        N,         Y,        N,       "bmodel"),
    "cv180x":    (N,   N,   Y,    Y,        N,         N,        N,       "cvimodel"),
    "cv181x":    (N,   N,   Y,    Y,        N,         N,        N,       "cvimodel"),
    "cv182x":    (N,   N,   Y,    Y,        N,         N,        N,       "cvimodel"),
    "cv183x":    (N,   N,   Y,    Y,        N,         N,        N,       "cvimodel"),
    "cv186x":    (Y,   Y,   Y,    Y,        N,         Y,        N,       "bmodel"),
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
    # model_name:              (bm1684, bm1684x, bm1686, cv180x, cv181x, cv182x, cv183x, cv186x)
    "mobilenet_v2_cf":            (N,      Y,       N,      Y,      N,      Y,      Y,     N),
    "resnet50_v2":                (Y,      N,       N,      N,      N,      Y,      Y,     N),
    "yolov5s":                    (N,      Y,       Y,      N,      N,      Y,      Y,     Y),
    "yolov5s_pt":                 (N,      Y,       N,      N,      N,      N,      N,     N),
    "yolov5s_tf":                 (N,      Y,       N,      N,      N,      N,      N,     N),
    "retinaface_mnet_with_det":   (N,      N,       N,      N,      Y,      Y,      Y,     N),
}


full_model_list = {
    # model_name:              (bm1684, bm1684x, bm1686, cv180x, cv181x, cv182x, cv183x, cv186x)
    ######## onnx ###############
    "bert-tiny_from_pt":          (N,      Y,       Y,      N,      N,      Y,      Y,      Y),
    "blazeface":                  (N,      Y,       N,      N,      N,      N,      Y,      N),
    "densenet121-12":             (N,      Y,       Y,      N,      N,      Y,      Y,      Y),
    "densenet201":                (N,      Y,       N,      N,      N,      N,      Y,      N),
    "ecanet50":                   (Y,      Y,       N,      N,      N,      N,      Y,      N),
    "efficientdet-d0":            (N,      Y,       N,      N,      N,      N,      Y,      N),
    "efficientnet":               (N,      Y,       Y,      N,      N,      Y,      Y,      Y),
    "inception_v3":               (N,      Y,       Y,      N,      N,      Y,      Y,      Y),
    "mnist-12":                   (N,      Y,       Y,      N,      N,      Y,      Y,      Y),
    "mobilenet_v2":               (N,      Y,       Y,      N,      N,      Y,      Y,      Y),
    "resnet18_v1":                (N,      Y,       Y,      N,      N,      Y,      Y,      Y),
    "resnet18_v2":                (N,      Y,       Y,      N,      N,      Y,      Y,      Y),
    "resnet50_v1":                (N,      Y,       Y,      N,      N,      Y,      Y,      Y),
    "resnet50_v2":                (Y,      Y,       Y,      N,      N,      Y,      Y,      Y),
    "res2net50":                  (N,      Y,       N,      N,      N,      N,      Y,      N),
    "retinaface":                 (N,      Y,       Y,      N,      N,      Y,      Y,      Y),
    "se-resnet50":                (N,      Y,       Y,      N,      N,      Y,      Y,      Y),
    "shufflenet_v2":              (N,      Y,       Y,      N,      N,      Y,      Y,      Y),
    "squeezenet1.0":              (N,      Y,       Y,      N,      N,      Y,      Y,      Y),
    "ssd-12":                     (N,      Y,       Y,      N,      N,      N,      N,      Y),
    "ultraface_640":              (N,      Y,       Y,      N,      N,      N,      N,      Y),
    "unet":                       (N,      Y,       N,      N,      N,      N,      Y,      N),
    "unet_plusplus":              (N,      N,       N,      N,      N,      N,      N,      N),
    "vgg16":                      (N,      Y,       Y,      N,      N,      Y,      Y,      Y),
    "yolov5s":                    (N,      Y,       Y,      N,      N,      Y,      Y,      Y),
    "yolov3":                     (N,      Y,       Y,      N,      N,      Y,      Y,      Y),
    "yolov3_tiny":                (N,      Y,       Y,      N,      N,      Y,      Y,      Y),
    "yolov4s":                    (N,      Y,       N,      N,      N,      N,      Y,      N),
    "yolox_s":                    (N,      Y,       Y,      N,      N,      Y,      Y,      Y),
    "yolov5s-face":               (N,      Y,       N,      N,      N,      N,      Y,      N),
    "yolov5s_with_trans":         (N,      Y,       Y,      N,      N,      Y,      Y,      Y),
    "alphapose_res50":            (N,      Y,       Y,      N,      N,      Y,      Y,      N),
    "yolov7":                     (N,      Y,       N,      N,      N,      N,      Y,      N),
    "yolov8s":                    (N,      Y,       N,      N,      N,      N,      N,      N),
    ######## Pytorch #######      ######
    "bert_pt":                    (N,      Y,       Y,      N,      N,      N,      N,      Y),
    "bert_base_pt":               (N,      N,       N,      N,      N,      N,      N,      N),
    "bert_large_pt":              (N,      N,       N,      N,      N,      N,      N,      N),
    "resnet50_pt":                (N,      Y,       Y,      N,      N,      Y,      Y,      Y),
    "sd_encoder_pt":              (N,      Y,       Y,      N,      N,      N,      N,      Y),
    "sd_decoder_pt":              (N,      Y,       Y,      N,      N,      N,      N,      Y),
    "yolov5s_pt":                 (N,      Y,       Y,      N,      N,      Y,      Y,      Y),
    ######## caffe #########      #####
    "mobilenet_v2_cf":            (N,      Y,       Y,      N,      N,      Y,      Y,      Y),
    "resnet18_cf":                (N,      Y,       Y,      N,      N,      Y,      Y,      Y),
    "segnet_cf":                  (N,      Y,       Y,      N,      N,      Y,      Y,      Y),
    "mobilenet_v2_cf_bs4":        (N,      Y,       N,      N,      N,      Y,      Y,      N),
    "retinaface_mnet_with_det":   (N,      Y,       N,      Y,      N,      Y,      Y,      N),
    "arcface_res50":              (N,      Y,       Y,      N,      N,      Y,      Y,      Y),
    "yolov3_416_with_det":        (N,      Y,       N,      N,      N,      Y,      Y,      N),
    "enet_cf":                    (N,      Y,       N,      N,      N,      N,      Y,      N),
    "erfnet_cf":                  (N,      Y,       N,      N,      N,      N,      Y,      N),
    "googlenet_cf":               (N,      Y,       N,      N,      N,      N,      Y,      N),
    "icnet_cf":                   (N,      N,       N,      N,      N,      N,      Y,      N),
    "inception_v4_cf":            (N,      Y,       N,      N,      N,      N,      Y,      N),
    "ssd300_cf":                  (N,      Y,       N,      N,      N,      N,      Y,      N),
    "yolov3_spp_cf":              (N,      Y,       N,      N,      N,      N,      Y,      N),
    "yolov4_cf":                  (N,      Y,       N,      N,      N,      N,      Y,      N),
    # ----- cvs20-test -----      -------
    "feature_extract_cf":         (N,      Y,       Y,      N,      N,      Y,      Y,      Y),
    "squeezenet_v1.1_cf_cvs20":   (N,      Y,       Y,      N,      N,      Y,      Y,      Y),
    "squeezenet_v1.1_cf":         (N,      Y,       Y,      N,      N,      Y,      Y,      Y),
    ######## tflite ########      ########
    "inception_v4_tf":            (N,      Y,       Y,      N,      N,      N,      N,      Y),
    "mobilenet_v2_tf":            (N,      Y,       Y,      N,      N,      N,      N,      Y),
    "resnet50_tf":                (N,      Y,       Y,      N,      N,      N,      N,      Y),
    "ssd_mobilenet_v1_tf":        (N,      Y,       Y,      N,      N,      N,      N,      Y),
    "yolov5s_tf":                 (N,      Y,       Y,      N,      N,      N,      N,      Y),
    "mobilebert_tf":              (N,      Y,       Y,      N,      N,      N,      N,      Y),
    ######## PaddlePaddle ##      ##############
    "pp_humanseg":                (N,      Y,       Y,      N,      N,      N,      N,      Y),
    "pp_liteseg":                 (N,      Y,       Y,      N,      N,      N,      N,      Y),
    "pp_picodet":                 (N,      Y,       Y,      N,      N,      Y,      Y,      Y),
    "pp_yoloe":                   (N,      Y,       Y,      N,      N,      Y,      Y,      Y),
    "pp_yolox":                   (N,      Y,       Y,      N,      N,      Y,      Y,      Y),
    "pp_yolov3":                  (N,      Y,       Y,      N,      N,      Y,      Y,      Y),
    "pp_ocr_det":                 (N,      Y,       Y,      N,      N,      Y,      Y,      Y),
    "pp_ocr_cls":                 (N,      Y,       Y,      N,      N,      Y,      Y,      Y),
    "pp_ocr_rec":                 (N,      Y,       Y,      N,      N,      N,      N,      Y),
    "pp_hgnet":                   (N,      Y,       Y,      N,      N,      Y,      Y,      Y),
}
