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
    # "mars3":     (Y,   Y,   Y,    Y,        N,         Y,        N,       "bmodel"),
    "cv180x":    (N,   N,   Y,    Y,        N,         N,        N,       "cvimodel"),
    "cv181x":    (N,   N,   Y,    Y,        N,         N,        N,       "cvimodel"),
    "cv182x":    (N,   N,   Y,    Y,        N,         N,        N,       "cvimodel"),
    "cv183x":    (N,   N,   Y,    Y,        N,         N,        N,       "cvimodel"),
    "cv186x":    (Y,   Y,   Y,    Y,        N,         Y,        N,       "bmodel"),
    "sg2260":    (Y,   Y,   Y,    N,        N,         Y,        N,       "bmodel"),
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
    # model_name:              (bm1684, bm1684x, bm1686, cv180x, cv181x, cv182x, cv183x, cv186x, sg2260)
    "mobilenet_v2_cf":            (N,      Y,       Y,      Y,      N,      Y,      Y,     N,      N),
    "resnet50_v2":                (Y,      N,       N,      N,      N,      Y,      Y,     N,      Y),
    "yolov5s":                    (N,      Y,       Y,      N,      N,      Y,      Y,     Y,      Y),
    "yolov5s_pt":                 (N,      Y,       N,      N,      N,      N,      N,     N,      N),
    "yolov5s_tf":                 (N,      Y,       Y,      N,      N,      N,      N,     N,      N),
    "retinaface_mnet_with_det":   (N,      N,       N,      N,      Y,      Y,      Y,     N,      N),
}


full_model_list = {
    # model_name:              (bm1684, bm1684x, bm1686, cv180x, cv181x, cv182x, cv183x, cv186x, sg2260)
    ######## onnx ###############
    "bert-tiny_from_pt":          (N,      Y,       Y,      N,      N,      Y,      Y,      Y,     Y),
    "blazeface":                  (N,      Y,       N,      Y,      N,      Y,      Y,      N,     N),
    "densenet121-12":             (N,      Y,       Y,      Y,      N,      Y,      Y,      Y,     N),
    "densenet201":                (N,      N,       N,      N,      N,      Y,      Y,      N,     N),
    "ecanet50":                   (N,      Y,       N,      N,      N,      N,      Y,      N,     N),
    "efficientdet-d0":            (N,      N,       N,      Y,      N,      Y,      Y,      N,     N),
    "efficientnet":               (N,      Y,       Y,      N,      N,      Y,      Y,      Y,     N),
    "inception_v3":               (N,      Y,       Y,      N,      N,      Y,      Y,      Y,     Y),
    "mnist-12":                   (N,      Y,       Y,      N,      N,      Y,      Y,      Y,     Y),
    "mobilenet_v2":               (N,      Y,       Y,      N,      N,      Y,      Y,      Y,     Y),
    "resnet18_v1":                (N,      Y,       Y,      Y,      N,      Y,      Y,      Y,     N),
    "resnet18_v2":                (N,      Y,       Y,      N,      N,      Y,      Y,      Y,     N),
    "resnet50_v1":                (N,      Y,       Y,      N,      N,      Y,      Y,      Y,     N),
    "resnet50_v2":                (Y,      Y,       Y,      N,      N,      Y,      Y,      Y,     N),
    "res2net50":                  (N,      Y,       N,      N,      N,      N,      Y,      N,     N),
    "retinaface":                 (N,      Y,       Y,      N,      N,      Y,      Y,      Y,     N),
    "se-resnet50":                (N,      Y,       Y,      N,      N,      Y,      Y,      Y,     N),
    "shufflenet_v2":              (N,      Y,       Y,      Y,      N,      Y,      Y,      Y,     Y),
    "squeezenet1.0":              (N,      Y,       Y,      N,      N,      Y,      Y,      Y,     Y),
    "ssd-12":                     (N,      Y,       Y,      N,      N,      N,      N,      Y,     Y),
    "ultraface_640":              (N,      Y,       Y,      N,      N,      N,      N,      Y,     N),
    "unet":                       (N,      N,       N,      N,      N,      N,      Y,      N,     N),
    "unet_plusplus":              (N,      N,       N,      N,      N,      N,      N,      N,     N),
    "vgg16":                      (N,      Y,       Y,      N,      N,      Y,      Y,      Y,     Y),
    "yolov5s":                    (N,      Y,       Y,      N,      N,      Y,      Y,      Y,     N),
    "yolov3":                     (N,      Y,       Y,      N,      N,      Y,      Y,      Y,     N),
    "yolov3_tiny":                (N,      Y,       Y,      N,      N,      Y,      Y,      Y,     Y),
    "yolov4s":                    (N,      Y,       N,      N,      N,      Y,      Y,      N,     N),
    "yolox_s":                    (N,      Y,       Y,      N,      N,      Y,      Y,      Y,     Y),
    "yolov5s-face":               (N,      Y,       N,      N,      N,      N,      Y,      N,     N),
    "yolov5s_with_trans":         (N,      Y,       Y,      N,      N,      Y,      Y,      Y,     N),
    "alphapose_res50":            (N,      Y,       Y,      N,      N,      Y,      Y,      N,     N),
    "yolov7":                     (N,      N,       N,      N,      N,      N,      Y,      N,     N),
    "yolov8s":                    (N,      N,       N,      N,      N,      N,      N,      N,     N),
    "yolov8n":                    (N,      N,       N,      N,      N,      Y,      Y,      N,     N),
    "nasnet_mobile":              (N,      N,       N,      Y,      N,      Y,      Y,      N,     N),
    "espcn_3x":                   (N,      N,       N,      Y,      N,      Y,      Y,      N,     N),
    ######## Pytorch #######      ######
    "bert_pt":                    (N,      Y,       Y,      N,      N,      N,      N,      Y,     Y),
    "bert_base_pt":               (N,      N,       N,      N,      N,      N,      N,      N,     N),
    "bert_large_pt":              (N,      N,       N,      N,      N,      N,      N,      N,     N),
    "resnet50_pt":                (N,      Y,       Y,      N,      N,      Y,      Y,      Y,     Y),
    "sd_encoder_pt":              (N,      Y,       Y,      N,      N,      N,      N,      Y,     N),
    "sd_decoder_pt":              (N,      Y,       Y,      N,      N,      N,      N,      Y,     N),
    "yolov5s_pt":                 (N,      Y,       Y,      N,      N,      Y,      Y,      Y,     N),
    ######## caffe #########      #####
    "mobilenet_v2_cf":            (N,      Y,       Y,      Y,      N,      Y,      Y,      Y,     Y),
    "resnet18_cf":                (N,      Y,       Y,      N,      N,      Y,      Y,      Y,     Y),
    "segnet_cf":                  (N,      Y,       Y,      N,      N,      Y,      Y,      Y,     Y),
    "mobilenet_v2_cf_bs4":        (N,      Y,       N,      N,      N,      Y,      Y,      N,     N),
    "retinaface_mnet_with_det":   (N,      N,       N,      Y,      N,      Y,      Y,      N,     N),
    "arcface_res50":              (N,      Y,       Y,      N,      N,      Y,      Y,      Y,     N),
    "yolov3_416_with_det":        (N,      N,       N,      N,      N,      Y,      Y,      N,     N),
    "enet_cf":                    (N,      N,       N,      N,      N,      N,      Y,      N,     N),
    "erfnet_cf":                  (N,      N,       N,      N,      N,      N,      Y,      N,     N),
    "googlenet_cf":               (N,      Y,       N,      Y,      N,      Y,      Y,      N,     N),
    "icnet_cf":                   (N,      N,       N,      N,      N,      N,      Y,      N,     N),
    "inception_v4_cf":            (N,      Y,       N,      N,      N,      N,      Y,      N,     N),
    "ssd300_cf":                  (N,      N,       N,      N,      N,      N,      Y,      N,     N),
    "yolov3_spp_cf":              (N,      N,       N,      N,      N,      N,      Y,      N,     N),
    "yolov4_cf":                  (N,      N,       N,      N,      N,      N,      Y,      N,     N),
    "resnext50_cf":               (N,      N,       N,      N,      N,      Y,      Y,      N,     N),
    "mobilenetv2_ssd_cf":         (N,      N,       N,      N,      N,      Y,      Y,      N,     N),
    # ----- cvs20-test -----      -------
    "feature_extract_cf":         (N,      Y,       Y,      N,      N,      Y,      Y,      Y,     N),
    "squeezenet_v1.1_cf_cvs20":   (N,      Y,       Y,      N,      N,      Y,      Y,      Y,     N),
    "squeezenet_v1.1_cf":         (N,      Y,       Y,      Y,      N,      Y,      Y,      Y,     N),
    ######## tflite ########      ########
    "inception_v4_tf":            (N,      Y,       Y,      N,      N,      N,      N,      Y,     N),
    "mobilenet_v2_tf":            (N,      Y,       Y,      N,      N,      N,      N,      Y,     N),
    "resnet50_tf":                (N,      Y,       Y,      N,      N,      N,      N,      Y,     N),
    "ssd_mobilenet_v1_tf":        (N,      Y,       Y,      N,      N,      N,      N,      Y,     N),
    "yolov5s_tf":                 (N,      Y,       Y,      N,      N,      N,      N,      Y,     N),
    "mobilebert_tf":              (N,      Y,       Y,      N,      N,      N,      N,      Y,     N),
    ######## PaddlePaddle ##      ##############
    "pp_humanseg":                (N,      Y,       Y,      N,      N,      N,      N,      Y,     N),
    "pp_liteseg":                 (N,      Y,       Y,      N,      N,      N,      N,      Y,     N),
    "pp_picodet":                 (N,      Y,       Y,      N,      N,      Y,      Y,      Y,     N),
    "pp_yoloe":                   (N,      Y,       Y,      N,      N,      Y,      Y,      Y,     N),
    "pp_yoloe_m":                 (N,      N,       N,      N,      N,      Y,      Y,      N,     N),
    "pp_yolox":                   (N,      Y,       Y,      N,      N,      Y,      Y,      Y,     N),
    "pp_yolov3":                  (N,      Y,       Y,      N,      N,      Y,      Y,      Y,     N),
    "pp_ocr_det":                 (N,      Y,       Y,      N,      N,      Y,      Y,      Y,     N),
    "pp_ocr_cls":                 (N,      Y,       Y,      N,      N,      Y,      Y,      Y,     N),
    "pp_ocr_rec":                 (N,      Y,       Y,      N,      N,      N,      N,      Y,     N),
    "pp_hgnet":                   (N,      Y,       Y,      N,      N,      Y,      Y,      Y,     N),
}
