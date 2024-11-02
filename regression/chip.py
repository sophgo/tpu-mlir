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
    # chip_name: (F32, F16, BF16, INT8_sym, INT8_asym, INT4_sym, f8e4m3, f8e5m2, dynamic, model_type)
    "bm1684":    (Y,   N,   N,    Y,        N,         N,        N,      N,      N,       "bmodel"),
    "bm1684x":   (Y,   Y,   Y,    Y,        Y,         N,        N,      N,      N,       "bmodel"),
    "bm1688":    (Y,   Y,   Y,    Y,        N,         Y,        N,      N,      N,       "bmodel"),
    "cv180x":    (N,   N,   Y,    Y,        N,         N,        N,      N,      N,       "cvimodel"),
    "cv181x":    (N,   N,   Y,    Y,        N,         N,        N,      N,      N,       "cvimodel"),
    "cv182x":    (N,   N,   Y,    Y,        N,         N,        N,      N,      N,       "cvimodel"),
    "cv183x":    (N,   N,   Y,    Y,        N,         N,        N,      N,      N,       "cvimodel"),
    "cv186x":    (Y,   Y,   Y,    Y,        N,         Y,        N,      N,      N,       "bmodel"),
    "bm1690":    (Y,   Y,   Y,    Y,        N,         Y,        N,      N,      N,       "bmodel"),
    "sg2380":    (Y,   Y,   Y,    Y,        N,         N,        N,      N,      N,       "bmodel"),
    "mars3":     (N,   N,   Y,    Y,        N,         N,        N,      N,      N,       "bmodel"),
}

multi_core_info = {
    # chip_name : num_core
    "bm1688": 2,
    "bm1690": 8,
    "sg2380": 4,
}

'''
    basic_model_list is for each commit test (int8_sym only)
    full_model_list is for daily test (all quant mode)

    Note:
        1. suffix: _pt = pytorch, _cf = caffe, _tf = tflite, default is onnx
        2. order of chips in model list should keep the same as in chip_support
        3. disable a certain quant mode in config file if needed
'''

######################################
# Model Support
######################################
basic_model_list = {
    # model_name:              (bm1684, bm1684x, bm1688, cv180x, cv181x, cv182x, cv183x, cv186x, bm1690, sg2380, mars3)
    "mobilenet_v2_cf":            (N,      Y,       Y,      Y,      N,      Y,      Y,     N,      N,       N,       Y),
    "resnet50_v2":                (Y,      N,       N,      N,      N,      Y,      Y,     N,      Y,       Y,       Y),
    "yolov5s":                    (N,      Y,       Y,      N,      N,      Y,      Y,     Y,      Y,       N,       Y),
    "yolov5s_pt":                 (N,      Y,       N,      N,      N,      N,      N,     N,      N,       N,       Y),
    "yolov5s_tf":                 (N,      Y,       Y,      N,      N,      N,      N,     N,      N,       N,       N),
    "retinaface_mnet_with_det":   (N,      N,       N,      N,      Y,      Y,      Y,     N,      N,       N,       N),
    "nmt_encode":                 (N,      N,       N,      N,      N,      Y,      Y,     N,      N,       N,       N),
    "nmt_decode10":               (N,      N,       N,      N,      N,      Y,      Y,     N,      N,       N,       N),
    "nmt_decode20":               (N,      N,       N,      N,      N,      Y,      Y,     N,      N,       N,       N),
    "bert-tiny_from_pt":          (N,      Y,       Y,      N,      N,      Y,      Y,     Y,      N,       N,       N),
    "eva02_block":                (N,      Y,       N,      N,      N,      N,      N,     N,      N,       N,       N),
    "swint_block":                (N,      Y,       Y,      N,      N,      N,      N,     N,      N,       N,       Y),
    "sam_block":                  (N,      Y,       N,      N,      N,      N,      N,     N,      N,       N,       N),
    "cswin_block":                (N,      Y,       N,      N,      N,      N,      N,     N,      N,       N,       Y),
}

full_model_list = {
    # model_name:              (bm1684, bm1684x, bm1688, cv180x, cv181x, cv182x, cv183x, cv186x, bm1690, sg2380, mars3)
    ######## onnx ###############
    "bert-tiny_from_pt":          (N,      Y,       Y,      N,      N,      Y,      Y,      Y,     N,     N,       N), # bm1690 int8_sym has problem
    "blazeface":                  (N,      Y,       N,      Y,      N,      Y,      Y,      N,     N,     N,       N),
    "densenet121-12":             (N,      Y,       Y,      Y,      N,      Y,      Y,      Y,     N,     N,       N),
    "densenet201":                (N,      N,       N,      N,      N,      Y,      Y,      N,     N,     N,       Y),
    "ecanet50":                   (N,      Y,       N,      N,      N,      N,      Y,      N,     N,     N,       N),
    "efficientdet-d0":            (N,      N,       N,      Y,      N,      Y,      Y,      N,     N,     N,       N),
    "efficientnet":               (N,      Y,       Y,      N,      N,      Y,      Y,      Y,     N,     N,       N),
    "efficientnetv2":             (N,      N,       N,      N,      N,      N,      N,      N,     N,     N,       Y),
    "inception_v3":               (N,      Y,       Y,      N,      N,      Y,      Y,      Y,     N,     N,       Y), # bm1690 f16, bf16 has problem
    "mnist-12":                   (N,      Y,       Y,      N,      N,      Y,      Y,      Y,     Y,     N,       N),
    "mobilenet_v2":               (N,      Y,       Y,      N,      N,      Y,      Y,      Y,     Y,     N,       Y),
    "resnet18_v1":                (N,      Y,       Y,      Y,      N,      Y,      Y,      Y,     Y,     N,       Y),
    "resnet18_v2":                (N,      Y,       Y,      N,      N,      Y,      Y,      Y,     N,     N,       Y),
    "resnet50_v1":                (N,      Y,       Y,      N,      N,      Y,      Y,      Y,     N,     N,       Y),
    "resnet50_v2":                (Y,      Y,       Y,      N,      N,      Y,      Y,      Y,     Y,     N,       N),
    "res2net50":                  (N,      Y,       N,      N,      N,      N,      Y,      N,     N,     N,       N),
    "retinaface":                 (N,      Y,       Y,      N,      N,      Y,      Y,      Y,     N,     N,       Y),
    "se-resnet50":                (N,      Y,       Y,      N,      N,      Y,      Y,      Y,     N,     N,       N),
    "shufflenet_v2":              (N,      Y,       Y,      Y,      N,      Y,      Y,      Y,     N,     N,       Y),
    "squeezenet1.0":              (N,      Y,       Y,      N,      N,      Y,      Y,      Y,     Y,     N,       Y),
    "ssd-12":                     (N,      Y,       Y,      N,      N,      N,      N,      Y,     Y,     N,       N),
    "swin_transformer":           (Y,      Y,       Y,      N,      N,      N,      N,      Y,     N,     N,       Y),
    "ultraface_640":              (N,      Y,       Y,      N,      N,      N,      N,      Y,     N,     N,       N),
    "unet":                       (N,      N,       N,      N,      N,      N,      Y,      N,     N,     N,       N),
    "unet_plusplus":              (N,      N,       N,      N,      N,      N,      N,      N,     N,     N,       N),
    "vgg16":                      (N,      Y,       Y,      N,      N,      Y,      Y,      Y,     Y,     N,       Y),
    "yolov5s":                    (N,      Y,       Y,      N,      N,      Y,      Y,      Y,     N,     N,       N),
    "yolov3":                     (N,      Y,       Y,      N,      N,      Y,      Y,      Y,     N,     N,       N),
    "yolov3_tiny":                (N,      Y,       Y,      N,      N,      Y,      Y,      Y,     Y,     N,       Y),
    "yolov4s":                    (N,      Y,       N,      N,      N,      Y,      Y,      N,     N,     N,       Y),
    "yolox_s":                    (N,      Y,       Y,      N,      N,      Y,      Y,      Y,     Y,     N,       N),
    "yolov5s-face":               (N,      Y,       N,      N,      N,      N,      Y,      N,     N,     N,       N),
    "yolov5s_with_trans":         (N,      Y,       Y,      N,      N,      Y,      Y,      Y,     N,     N,       N),
    "alphapose_res50":            (N,      Y,       Y,      N,      N,      Y,      Y,      N,     N,     N,       N),
    "yolov7":                     (N,      N,       N,      N,      N,      N,      Y,      N,     N,     N,       N),
    "yolov8s":                    (N,      N,       N,      N,      N,      N,      N,      N,     N,     N,       Y),
    "yolov8n":                    (N,      N,       N,      N,      N,      Y,      Y,      N,     N,     N,       N),
    "nasnet_mobile":              (N,      N,       N,      Y,      N,      Y,      Y,      N,     N,     N,       N),
    "espcn_3x":                   (N,      N,       N,      Y,      N,      Y,      Y,      N,     N,     N,       N),
    "nmt_encode":                 (N,      N,       N,      N,      N,      Y,      Y,      N,     N,     N,       N),
    "nmt_decode10":               (N,      N,       N,      N,      N,      Y,      Y,      N,     N,     N,       N),
    "nmt_decode20":               (N,      N,       N,      N,      N,      Y,      Y,      N,     N,     N,       N),
    ######## Pytorch #######      ######
    "bert_pt":                    (N,      Y,       Y,      N,      N,      N,      N,      Y,     Y,     N,       N),
    "bert_base_pt":               (N,      N,       N,      N,      N,      N,      N,      N,     N,     N,       N),
    "bert_large_pt":              (N,      N,       N,      N,      N,      N,      N,      N,     N,     N,       N),
    "resnet50_pt":                (N,      Y,       Y,      N,      N,      Y,      Y,      Y,     Y,     N,       N),
    "sd_encoder_pt":              (N,      Y,       Y,      N,      N,      N,      N,      Y,     N,     N,       N),
    "sd_decoder_pt":              (N,      N,       N,      N,      N,      N,      N,      N,     N,     N,       N), # timeout
    "yolov5s_pt":                 (N,      Y,       Y,      N,      N,      Y,      Y,      Y,     N,     N,       N),
    "resnext_pt":                 (N,      N,       N,      N,      N,      N,      N,      N,     N,     N,       Y),
    ######## caffe #########      #####
    "mobilenet_v2_cf":            (N,      Y,       Y,      Y,      N,      Y,      Y,      Y,     Y,     N,       N),
    "resnet18_cf":                (N,      Y,       Y,      N,      N,      Y,      Y,      Y,     Y,     N,       N),
    "segnet_cf":                  (N,      Y,       Y,      N,      N,      Y,      Y,      Y,     Y,     N,       N),
    "mobilenet_v2_cf_bs4":        (N,      Y,       N,      N,      N,      Y,      Y,      N,     N,     N,       N),
    "retinaface_mnet_with_det":   (N,      N,       N,      Y,      N,      Y,      Y,      N,     N,     N,       N),
    "arcface_res50":              (N,      Y,       Y,      N,      N,      Y,      Y,      Y,     N,     N,       Y),
    "yolov3_416_with_det":        (N,      N,       N,      N,      N,      Y,      Y,      N,     N,     N,       N),
    "enet_cf":                    (N,      N,       N,      N,      N,      N,      Y,      N,     N,     N,       N),
    "erfnet_cf":                  (N,      N,       N,      N,      N,      N,      Y,      N,     N,     N,       N),
    "googlenet_cf":               (N,      Y,       N,      Y,      N,      Y,      Y,      N,     N,     N,       Y),
    "icnet_cf":                   (N,      N,       N,      N,      N,      N,      Y,      N,     N,     N,       N),
    "inception_v4_cf":            (N,      Y,       N,      N,      N,      N,      Y,      N,     N,     N,       Y),
    "ssd300_cf":                  (N,      N,       N,      N,      N,      N,      Y,      N,     N,     N,       N),
    "yolov3_spp_cf":              (N,      N,       N,      N,      N,      N,      Y,      N,     N,     N,       N),
    "yolov4_cf":                  (N,      N,       N,      N,      N,      N,      Y,      N,     N,     N,       N),
    "resnext50_cf":               (N,      N,       N,      N,      N,      Y,      Y,      N,     N,     N,       N),
    "mobilenetv2_ssd_cf":         (N,      N,       N,      N,      N,      Y,      Y,      N,     N,     N,       N),
    # ----- cvs20-test -----      -------
    "feature_extract_cf":         (N,      Y,       Y,      N,      N,      Y,      Y,      Y,     N,     N,       N),
    "squeezenet_v1.1_cf_cvs20":   (N,      Y,       Y,      N,      N,      Y,      Y,      Y,     Y,     N,       Y),
    "squeezenet_v1.1_cf":         (N,      Y,       Y,      Y,      N,      Y,      Y,      Y,     N,     N,       N),
    ######## tflite ########      ########
    "inception_v4_tf":            (N,      Y,       Y,      N,      N,      N,      N,      Y,     N,     N,       N),
    "mobilenet_v2_tf":            (N,      Y,       Y,      N,      N,      N,      N,      Y,     N,     N,       N),
    "resnet50_tf":                (N,      Y,       Y,      N,      N,      N,      N,      Y,     N,     N,       N),
    "ssd_mobilenet_v1_tf":        (N,      Y,       Y,      N,      N,      N,      N,      Y,     N,     N,       N),
    "yolov5s_tf":                 (N,      Y,       Y,      N,      N,      N,      N,      Y,     N,     N,       N),
    "mobilebert_tf":              (N,      Y,       Y,      N,      N,      N,      N,      Y,     N,     N,       N),
    ######## PaddlePaddle ##      ##############
    "pp_humanseg":                (N,      Y,       Y,      N,      N,      N,      N,      Y,     N,     N,       N),
    "pp_liteseg":                 (N,      Y,       Y,      N,      N,      N,      N,      Y,     N,     N,       N),
    "pp_picodet":                 (N,      Y,       Y,      N,      N,      Y,      Y,      Y,     N,     N,       N),
    "pp_yoloe":                   (N,      Y,       Y,      N,      N,      Y,      Y,      Y,     N,     N,       Y),
    "pp_yoloe_m":                 (N,      N,       N,      N,      N,      Y,      Y,      N,     N,     N,       N),
    "pp_yolox":                   (N,      Y,       Y,      N,      N,      Y,      Y,      Y,     N,     N,       N),
    "pp_yolov3":                  (N,      Y,       Y,      N,      N,      Y,      Y,      Y,     N,     N,       N),
    "pp_ocr_det":                 (N,      Y,       Y,      N,      N,      Y,      Y,      Y,     N,     N,       Y),
    "pp_ocr_cls":                 (N,      Y,       Y,      N,      N,      Y,      Y,      Y,     N,     N,       Y),
    "pp_ocr_rec":                 (N,      Y,       N,      N,      N,      N,      N,      N,     N,     N,       Y), # bf16 bm1688, cv186x has problem
    "pp_hgnet":                   (N,      Y,       Y,      N,      N,      Y,      Y,      Y,     N,     N,       N),
}

basic_multi_core_model_list = {
    # model_name:                 (bm1684, bm1684x, bm1688, cv180x, cv181x, cv182x, cv183x, cv186x, bm1690, sg2380, mars3)
    ######## onnx ###############
    "attention_2batch_block_0_1": (N,      N,       Y,      N,      N,      N,      N,      N,      Y,         N,       N),
    "pp_yoloe":                   (N,      N,       Y,      N,      N,      N,      N,      N,      Y,         N,       N),
    "inception_v3":               (N,      N,       Y,      N,      N,      N,      N,      N,      Y,         N,       N),
    "pp_ocr_cls":                 (N,      N,       Y,      N,      N,      N,      N,      N,      Y,         N,       N),
}

full_multi_core_model_list = {
    # model_name:                 (bm1684, bm1684x, bm1688, cv180x, cv181x, cv182x, cv183x, cv186x, bm1690, sg2380, mars3)
    ###, sg2380##### onnx ###############
    "attention_2batch_block_0_1": (N,      N,       Y,      N,      N,      N,      N,      N,      Y,       N,       N),
    "pp_yoloe":                   (N,      N,       Y,      N,      N,      N,      N,      N,      Y,       N,       N),
    "inception_v3":               (N,      N,       Y,      N,      N,      N,      N,      N,      N,       N,       N), # bm1690 f16, b16 has problem
    "pp_ocr_cls":                 (N,      N,       Y,      N,      N,      N,      N,      N,      Y,       N,       N),
}
