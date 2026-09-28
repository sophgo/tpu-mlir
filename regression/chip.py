#!/usr/bin/env python3
# Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
#
# TPU-MLIR is licensed under the 2-Clause BSD License except for the
# third-party components.
#
# ==============================================================================
"""Chip- and model-support matrices used by the regression driver.

Each entry in :data:`basic_model_list` / :data:`full_model_list` /
:data:`basic_multi_core_model_list` / :data:`full_multi_core_model_list`
is keyed by model name and holds a tuple of booleans whose order matches the
keys of :data:`chip_support` (also exposed as :data:`CHIP_COLUMNS`). Use
:func:`models_for_chip` to extract the models enabled for a given chip
without indexing the tuple manually.
"""

# yapf: disable
######################################
# Chip Support
######################################
Y, N = True, False

chip_support = {
    # chip_name: (F32, F16, BF16, INT8_sym, INT8_asym, INT4_sym, f8e4m3, f8e5m2, dynamic, model_type)
    "bm1684x":   (Y,   Y,   Y,    Y,        Y,         N,        N,      N,      N,       "bmodel"),
    "cv84x6":  (Y,   Y,   Y,    Y,        Y,         N,        N,      N,      N,       "bmodel"),
    "bm1688":    (Y,   Y,   Y,    Y,        N,         Y,        N,      N,      N,       "bmodel"),
    "cv190x":   (N,   N,   Y,    Y,        N,         N,        N,      N,      N,       "cvimodel"),
    "cv180x":    (N,   N,   Y,    Y,        N,         N,        N,      N,      N,       "cvimodel"),
    "cv181x":    (N,   N,   Y,    Y,        N,         N,        N,      N,      N,       "cvimodel"),
    "cv182x":    (N,   N,   Y,    Y,        N,         N,        N,      N,      N,       "cvimodel"),
    "cv183x":    (N,   N,   Y,    Y,        N,         N,        N,      N,      N,       "cvimodel"),
    "cv186x":    (Y,   Y,   Y,    Y,        N,         Y,        N,      N,      N,       "bmodel"),
    "bm1690":    (Y,   Y,   Y,    Y,        N,         Y,        N,      N,      N,       "bmodel"),
    "bm1690e":   (Y,   Y,   Y,    Y,        N,         Y,        N,      N,      N,       "bmodel"),
    "sg2380":    (N,   N,   N,    N,        N,         N,        N,      N,      N,       "bmodel"), # sg2380 no longer in use
    "cv184x":    (N,   N,   Y,    Y,        N,         N,        N,      N,      N,       "bmodel"),
    "sgtpuv8":   (N,   N,   Y,    Y,        N,         N,        N,      N,      N,       "bmodel"),
}

multi_core_info = {
    # chip_name : num_core
    "bm1688": 2,
    "bm1690": 8,
    "bm1690e": 4,
    "sg2380": 4,
    "cv84x6": 4,
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
    # model_name:              (bm1684x, cv84x6, bm1688, cv180x, cv181x, cv182x, cv183x, cv186x, bm1690, sg2380, cv184x, sgtpuv8, bm1690e)
    "mobilenet_v2_cf":            (Y,       Y,       Y,      Y,      N,      Y,      Y,      N,      N,      N,      Y,      Y,       N),
    "resnet50_v2":                (N,       N,       N,      N,      N,      Y,      Y,      N,      Y,      N,      Y,      Y,       Y),
    "yolov5s":                    (Y,       Y,       Y,      N,      N,      Y,      Y,      Y,      Y,      N,      Y,      Y,       Y),
    "yolov5s_pt":                 (Y,       Y,       N,      N,      N,      N,      N,      N,      N,      N,      Y,      N,       N),
    "yolov5s_tf":                 (N,       N,       N,      N,      N,      N,      N,      N,      N,      N,      N,      N,       N),
    "retinaface_mnet_with_det":   (N,       N,       N,      N,      Y,      Y,      Y,      N,      N,      N,      N,      N,       N),
    "nmt_encode":                 (N,       N,       N,      N,      N,      Y,      Y,      N,      N,      N,      N,      N,       N),
    "nmt_decode10":               (N,       N,       N,      N,      N,      Y,      Y,      N,      N,      N,      N,      N,       N),
    "nmt_decode20":               (N,       N,       N,      N,      N,      Y,      Y,      N,      N,      N,      N,      N,       N),
    "bert-tiny_from_pt":          (Y,       Y,       Y,      N,      N,      Y,      Y,      Y,      N,      N,      N,      N,       N),
    "eva02_block":                (Y,       Y,       N,      N,      N,      N,      N,      N,      N,      N,      N,      N,       N),
    "swint_block":                (Y,       Y,       Y,      N,      N,      N,      N,      N,      N,      N,      Y,      N,       N),
    "sam_block":                  (Y,       Y,       N,      N,      N,      N,      N,      N,      N,      N,      N,      N,       N),
    "cswin_block":                (Y,       Y,       N,      N,      N,      N,      N,      N,      N,      N,      Y,      N,       N),
}

full_model_list = {
    # model_name:              (bm1684x, cv84x6, bm1688, cv180x, cv181x, cv182x, cv183x, cv186x, bm1690, sg2380, cv184x, sgtpuv8, bm1690e)
    ######## onnx ###############
    "bert-tiny_from_pt":          (Y,       Y,       Y,      N,      N,      Y,      Y,      Y,      N,      N,      N,      N,       N), # bm1690 int8_sym has problem
    "blazeface":                  (Y,       Y,       N,      Y,      N,      Y,      Y,      N,      N,      N,      N,      N,       N),
    "densenet121-12":             (Y,       Y,       Y,      Y,      N,      Y,      Y,      Y,      N,      N,      N,      N,       N),
    "densenet201":                (N,       N,       N,      N,      N,      Y,      Y,      N,      N,      N,      Y,      N,       N),
    "ecanet50":                   (Y,       Y,       N,      N,      N,      N,      Y,      N,      N,      N,      N,      N,       N),
    "efficientdet-d0":            (N,       N,       N,      Y,      N,      Y,      Y,      N,      N,      N,      N,      N,       N),
    "efficientnet":               (Y,       Y,       Y,      N,      N,      Y,      Y,      Y,      N,      N,      N,      N,       N),
    "efficientnetv2":             (N,       N,       N,      N,      N,      N,      N,      N,      N,      N,      Y,      N,       N),
    "inception_v3":               (Y,       Y,       Y,      N,      N,      Y,      Y,      Y,      N,      N,      Y,      N,       N), # bm1690 f16, bf16 has problem
    "mnist-12":                   (Y,       Y,       Y,      N,      N,      Y,      Y,      Y,      Y,      N,      N,      N,       N),
    "mobilenet_v2":               (Y,       Y,       Y,      N,      N,      Y,      Y,      Y,      Y,      N,      Y,      N,       N),
    "resnet18_v1":                (Y,       Y,       Y,      Y,      N,      Y,      Y,      Y,      Y,      N,      Y,      N,       N),
    "resnet18_v2":                (Y,       Y,       Y,      N,      N,      Y,      Y,      Y,      N,      N,      Y,      N,       N),
    "resnet50_v1":                (Y,       Y,       Y,      N,      N,      Y,      Y,      Y,      N,      N,      Y,      N,       N),
    "resnet50_v2":                (Y,       Y,       Y,      N,      N,      Y,      Y,      Y,      Y,      N,      N,      N,       N),
    "res2net50":                  (Y,       Y,       N,      N,      N,      N,      Y,      N,      N,      N,      N,      N,       N),
    "retinaface":                 (Y,       Y,       Y,      N,      N,      Y,      Y,      Y,      N,      N,      Y,      N,       N),
    "se-resnet50":                (Y,       Y,       Y,      N,      N,      Y,      Y,      Y,      N,      N,      N,      N,       N),
    "shufflenet_v2":              (Y,       Y,       Y,      Y,      N,      Y,      Y,      Y,      N,      N,      Y,      N,       N),
    "squeezenet1.0":              (Y,       Y,       Y,      N,      N,      Y,      Y,      Y,      Y,      N,      Y,      Y,       N),
    "ssd-12":                     (Y,       Y,       Y,      N,      N,      N,      N,      Y,      Y,      N,      N,      N,       N),
    "swin_transformer":           (Y,       Y,       Y,      N,      N,      N,      N,      Y,      N,      N,      Y,      N,       N),
    "ultraface_640":              (Y,       Y,       Y,      N,      N,      N,      N,      Y,      N,      N,      N,      N,       N),
    "unet":                       (N,       N,       N,      N,      N,      N,      Y,      N,      N,      N,      N,      N,       N),
    "unet_plusplus":              (N,       N,       N,      N,      N,      N,      N,      N,      N,      N,      N,      N,       N),
    "vgg16":                      (Y,       Y,       Y,      N,      N,      Y,      Y,      Y,      Y,      N,      Y,      N,       N),
    "yolov5s":                    (Y,       Y,       Y,      N,      N,      Y,      Y,      Y,      N,      N,      N,      N,       N),
    "yolov3":                     (Y,       Y,       Y,      N,      N,      Y,      Y,      Y,      N,      N,      N,      N,       N),
    "yolov3_tiny":                (Y,       Y,       Y,      N,      N,      Y,      Y,      Y,      Y,      N,      Y,      N,       N),
    "yolov4s":                    (Y,       Y,       N,      N,      N,      Y,      Y,      N,      N,      N,      Y,      N,       N),
    "yolox_s":                    (Y,       Y,       Y,      N,      N,      Y,      Y,      Y,      Y,      N,      N,      N,       N),
    "yolov5s-face":               (Y,       Y,       N,      N,      N,      N,      Y,      N,      N,      N,      N,      N,       N),
    "yolov5s_with_trans":         (Y,       Y,       Y,      N,      N,      Y,      Y,      Y,      N,      N,      N,      N,       N),
    "alphapose_res50":            (Y,       Y,       Y,      N,      N,      Y,      Y,      N,      N,      N,      N,      N,       N),
    "yolov7":                     (N,       N,       N,      N,      N,      N,      Y,      N,      N,      N,      N,      N,       N),
    "yolov8s":                    (N,       N,       N,      N,      N,      N,      N,      N,      N,      N,      Y,      N,       N),
    "yolov8n":                    (N,       N,       N,      N,      N,      Y,      Y,      N,      N,      N,      N,      N,       N),
    "nasnet_mobile":              (N,       N,       N,      Y,      N,      Y,      Y,      N,      N,      N,      N,      N,       N),
    "espcn_3x":                   (N,       N,       N,      Y,      N,      Y,      Y,      N,      N,      N,      N,      N,       N),
    "nmt_encode":                 (N,       N,       N,      N,      N,      Y,      Y,      N,      N,      N,      N,      N,       N),
    "nmt_decode10":               (N,       N,       N,      N,      N,      Y,      Y,      N,      N,      N,      N,      N,       N),
    "nmt_decode20":               (N,       N,       N,      N,      N,      Y,      Y,      N,      N,      N,      N,      N,       N),
    ######## Pytorch #######      ######
    "bert_pt":                    (Y,       Y,       Y,      N,      N,      N,      N,      Y,      Y,      N,      N,      N,       N),
    "bert_base_pt":               (N,       N,       N,      N,      N,      N,      N,      N,      N,      N,      N,      N,       N),
    "bert_large_pt":              (N,       N,       N,      N,      N,      N,      N,      N,      N,      N,      N,      N,       N),
    "resnet50_pt":                (Y,       Y,       Y,      N,      N,      Y,      Y,      Y,      Y,      N,      N,      N,       N),
    "sd_encoder_pt":              (Y,       Y,       Y,      N,      N,      N,      N,      Y,      N,      N,      N,      N,       N),
    "sd_decoder_pt":              (N,       N,       N,      N,      N,      N,      N,      N,      N,      N,      N,      N,       N), # timeout
    "yolov5s_pt":                 (Y,       Y,       Y,      N,      N,      Y,      Y,      Y,      N,      N,      N,      N,       N),
    "resnext_pt":                 (N,       N,       N,      N,      N,      N,      N,      N,      N,      N,      Y,      N,       N),
    ######## caffe #########      #####
    "mobilenet_v2_cf":            (Y,       Y,       Y,      Y,      N,      Y,      Y,      Y,      Y,      N,      N,      N,       N),
    "resnet18_cf":                (Y,       Y,       Y,      N,      N,      Y,      Y,      Y,      Y,      N,      N,      N,       N),
    "segnet_cf":                  (Y,       Y,       Y,      N,      N,      Y,      Y,      Y,      Y,      N,      N,      N,       N),
    "mobilenet_v2_cf_bs4":        (Y,       Y,       N,      N,      N,      Y,      Y,      N,      N,      N,      N,      N,       N),
    "retinaface_mnet_with_det":   (N,       N,       N,      Y,      N,      Y,      Y,      N,      N,      N,      N,      N,       N),
    "arcface_res50":              (Y,       Y,       Y,      N,      N,      Y,      Y,      Y,      N,      N,      Y,      N,       N),
    "yolov3_416_with_det":        (N,       N,       N,      N,      N,      Y,      Y,      N,      N,      N,      N,      N,       N),
    "enet_cf":                    (N,       N,       N,      N,      N,      N,      Y,      N,      N,      N,      N,      N,       N),
    "erfnet_cf":                  (N,       N,       N,      N,      N,      N,      Y,      N,      N,      N,      N,      N,       N),
    "googlenet_cf":               (Y,       Y,       N,      Y,      N,      Y,      Y,      N,      N,      N,      Y,      N,       N),
    "icnet_cf":                   (N,       N,       N,      N,      N,      N,      Y,      N,      N,      N,      N,      N,       N),
    "inception_v4_cf":            (Y,       Y,       N,      N,      N,      N,      Y,      N,      N,      N,      Y,      N,       N),
    "ssd300_cf":                  (N,       N,       N,      N,      N,      N,      Y,      N,      N,      N,      N,      N,       N),
    "yolov3_spp_cf":              (N,       N,       N,      N,      N,      N,      Y,      N,      N,      N,      N,      N,       N),
    "yolov4_cf":                  (N,       N,       N,      N,      N,      N,      Y,      N,      N,      N,      N,      N,       N),
    "resnext50_cf":               (N,       N,       N,      N,      N,      Y,      Y,      N,      N,      N,      N,      N,       N),
    "mobilenetv2_ssd_cf":         (N,       N,       N,      N,      N,      Y,      Y,      N,      N,      N,      N,      N,       N),
    # ----- cvs20-test -----      -------
    "feature_extract_cf":         (Y,       Y,       Y,      N,      N,      Y,      Y,      Y,      N,      N,      N,      N,       N),
    "squeezenet_v1.1_cf_cvs20":   (Y,       Y,       Y,      N,      N,      Y,      Y,      Y,      Y,      N,      Y,      N,       N),
    "squeezenet_v1.1_cf":         (Y,       Y,       Y,      Y,      N,      Y,      Y,      Y,      N,      N,      N,      N,       N),
    ######## tflite ########      ########
    "inception_v4_tf":            (N,       N,       N,      N,      N,      N,      N,      N,      N,      N,      N,      N,       N),
    "mobilenet_v2_tf":            (N,       N,       N,      N,      N,      N,      N,      N,      N,      N,      N,      N,       N),
    "resnet50_tf":                (N,       N,       N,      N,      N,      N,      N,      N,      N,      N,      N,      N,       N),
    "ssd_mobilenet_v1_tf":        (N,       N,       N,      N,      N,      N,      N,      N,      N,      N,      N,      N,       N),
    "yolov5s_tf":                 (N,       N,       N,      N,      N,      N,      N,      N,      N,      N,      N,      N,       N),
    "mobilebert_tf":              (N,       N,       N,      N,      N,      N,      N,      N,      N,      N,      N,      N,       N),
    ######## PaddlePaddle ##      ##############
    "pp_humanseg":                (Y,       Y,       Y,      N,      N,      N,      N,      Y,      N,      N,      N,      N,       N),
    "pp_liteseg":                 (Y,       Y,       Y,      N,      N,      N,      N,      Y,      N,      N,      N,      N,       N),
    "pp_picodet":                 (Y,       Y,       Y,      N,      N,      Y,      Y,      Y,      N,      N,      N,      N,       N),
    "pp_yoloe":                   (Y,       Y,       Y,      N,      N,      Y,      Y,      Y,      N,      N,      Y,      N,       N),
    "pp_yoloe_m":                 (N,       N,       N,      N,      N,      Y,      Y,      N,      N,      N,      N,      N,       N),
    "pp_yolox":                   (Y,       Y,       Y,      N,      N,      Y,      Y,      Y,      N,      N,      N,      N,       N),
    "pp_yolov3":                  (Y,       Y,       Y,      N,      N,      Y,      Y,      Y,      N,      N,      N,      N,       N),
    "pp_ocr_det":                 (Y,       Y,       Y,      N,      N,      Y,      Y,      Y,      N,      N,      Y,      N,       N),
    "pp_ocr_cls":                 (Y,       Y,       Y,      N,      N,      Y,      Y,      Y,      N,      N,      Y,      N,       N),
    "pp_ocr_rec":                 (Y,       Y,       N,      N,      N,      N,      N,      N,      N,      N,      Y,      N,       N), # bf16 bm1688, cv186x has problem
    "pp_hgnet":                   (Y,       Y,       Y,      N,      N,      Y,      Y,      Y,      N,      N,      N,      N,       N),
}

basic_multi_core_model_list = {
    # model_name:              (bm1684x, cv84x6, bm1688, cv180x, cv181x, cv182x, cv183x, cv186x, bm1690, sg2380, cv184x, sgtpuv8, bm1690e)
    ######## onnx ###############
    "attention_2batch_block_0_1": (N,       Y,       Y,      N,      N,      N,      N,      N,      Y,      N,      N,      N,       N),
    "pp_yoloe":                   (N,       Y,       Y,      N,      N,      N,      N,      N,      Y,      N,      N,      N,       N),
    "inception_v3":               (N,       Y,       Y,      N,      N,      N,      N,      N,      Y,      N,      N,      N,       N),
    "pp_ocr_cls":                 (N,       Y,       Y,      N,      N,      N,      N,      N,      Y,      N,      N,      N,       N),
}

full_multi_core_model_list = {
    # model_name:              (bm1684x, cv84x6, bm1688, cv180x, cv181x, cv182x, cv183x, cv186x, bm1690, sg2380, cv184x, sgtpuv8, bm1690e)
    ###, sg2380##### onnx ###############
    "attention_2batch_block_0_1": (N,       Y,       Y,      N,      N,      N,      N,      N,      Y,      N,      N,      N,       N),
    "pp_yoloe":                   (N,       Y,       Y,      N,      N,      N,      N,      N,      Y,      N,      N,      N,       N),
    "inception_v3":               (N,       Y,       Y,      N,      N,      N,      N,      N,      N,      N,      N,      N,       N), # bm1690 f16, b16 has problem
    "pp_ocr_cls":                 (N,       Y,       Y,      N,      N,      N,      N,      N,      Y,      N,      N,      N,       N),
}

# yapf: enable

######################################
# Helpers
######################################
# Ordered list of chip names — also defines the column order used by every
# ``*_model_list`` tuple above. Iterating ``CHIP_COLUMNS`` (or
# ``chip_support.keys()``) is the canonical way to map a tuple index to a chip.
CHIP_COLUMNS = tuple(chip_support.keys())


def models_for_chip(model_list, chip):
    """Return the names of models in ``model_list`` enabled for ``chip``.

    ``model_list`` must use :data:`CHIP_COLUMNS` ordering. Returns ``[]`` for
    unknown chips, which silently filters out columns that have been added to
    ``chip_support`` but not yet to a particular model list.
    """
    try:
        idx = CHIP_COLUMNS.index(chip)
    except ValueError:
        return []
    return [name for name, flags in model_list.items() if idx < len(flags) and flags[idx]]
