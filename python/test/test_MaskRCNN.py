#!/usr/bin/env python3
# Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
#
# TPU-MLIR is licensed under the 2-Clause BSD License except for the
# third-party components.
#
# ==============================================================================
import numpy as np
from typing import List, Union

from tools.model_transform import *
from tools.tool_maskrcnn import MaskRCNN_InputPreprocessor, MaskRCNN_Tester_Basic
from utils.mlir_shell import *
from utils.mlir_shell import _os_system
from utils.auto_remove import clean_kmp_files
from utils.timer import Timer
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.jit as jit
import traceback

import yaml
import argparse
import pathlib


class Desc():

    def __init__(self, dtype, min=-10, max=10) -> None:
        self.dtype = dtype
        self.min = min
        self.max = max


class MaskRCNN_IR_TESTER(MaskRCNN_Tester_Basic):
    ID = 0
    CURRENT_CASE = ""
    dict_MaskRCNN_packed_interface = [
        "maskrcnn_output_num", "maskrcnn_structure", "maskrcnn_ppl_op",
        "numPPLOp_InWithoutWeight_MaskRCNN", "maskrcnn_io_map"
    ]
    dict_MaskRCNN_packed_interface = [
        "model_def", "model_extern", "model_name", "input_shapes", "test_input", "test_reference"
    ] + dict_MaskRCNN_packed_interface

    def __init__(self,
                 model_def_multi_torch: list[str] = None,
                 chip: str = "bm1684x",
                 mode: str = "all",
                 debug: bool = False,
                 num_core: int = 1,
                 debug_cmd: str = '',
                 path_yaml: str = None,
                 basic_max_shape_inverse: list[int] = [1216, 800],
                 basic_scalar_factor: list[float] = [1.8734375, 1.8735363],
                 path_custom_dataset: str = None):
        MaskRCNN_Tester_Basic.__init__(self, debug, path_custom_dataset)
        self.path_custom_dataset = path_custom_dataset
        self.data_npz_path_init()
        Y, N = True, False
        self.model_def_multi_torch = model_def_multi_torch
        self.debug = debug
        self.debug_cmd = debug_cmd
        # yapf: disable
        self.test_cases = {
            ##################################
            # MaskRCNN Test Case, Topologically
            ##################################
            # case: (test, bm1684x_support)
            #"MaskRCNN_Utest_SwinTBackBone":  (self.test_MaskRCNN_Utest_SwintT_BackBone,  N), # no torch.pt, it's on model-zoo
            "MaskRCNN_Utest_RPNGetBboxes":   (self.test_MaskRCNN_Utest_RPNGetBboxes,     Y),
            "MaskRCNN_Utest_BboxPooler":     (self.test_MaskRCNN_Utest_BboxPooler,       Y),
            #"MaskRCNN_Utest_GetBboxB":       (self.test_MaskRCNN_Utest_GetBboxB,         Y),
            "MaskRCNN_Utest_MaskPooler":     (self.test_MaskRCNN_Utest_MaskPooler,       Y),
            #"MaskRCNN_RPN_to_BboxPooler":    (self.test_MaskRCNN_RPN_to_BboxPooler,      N), #long time
            #"MaskRCNN_End2End":              (self.test_MaskRCNN_End2End,                N)  #long time
        }

        # yapf: enable

        self.model_file = ".bmodel"
        self.chip = chip.lower()
        self.num_core = num_core
        assert self.num_core == 1

        self.mode = mode.lower()
        self.path_yaml = path_yaml
        if not os.path.exists(self.path_default_MaskRCNN_dataset):
            assert 0, "[MaskRCNN-Error]  for test_MaskRCNN.py npz/pt is loaded in only one way, 1st is NNMODELS_PATH for regression,please ensure {} exists, model-zoo never use this file!".format(
                self.path_default_MaskRCNN_dataset)
        if self.path_yaml is None:
            self.path_yaml = self.path_default_MaskRCNN_dataset + "CONFIG_MaskRCNN.yaml"

        assert os.path.exists(
            self.path_yaml
        ), "[MaskRCNN-Test-Error] {} yaml path is not exist! There's example at tpu-mlir/regression/dataset/MaskRCNN/CONFIG_MaskRCNN.yaml".format(
            self.path_yaml, )

        self.basic_max_shape_inverse = basic_max_shape_inverse
        self.basic_scalar_factor = basic_scalar_factor
        self.info_helper()

    '''
        print necessary helper
    '''

    def info_helper(self):
        self.print_debug(
            "-----------------Welcome Use Superior MaskRCNN Test Assistant------------")
        self.print_debug(
            "[MaskRCNN-Help-0] default config yaml at path:{}; but utest will just copy it to their files!",
            self.path_default_MaskRCNN_dataset)
        self.print_debug(
            "[MaskRCNN-Help-1] mlir_transform external paramter will inject into the copy config yaml automatically"
        )
        self.print_debug(
            "[MaskRCNN-Help-2] default dataset is at your nnmodels, right now is at  {}, you can revise it use path_custom_dataset, which is {}, maybe you have changed it!"
            .format(self.path_default_MaskRCNN_dataset, self.path_custom_dataset))
        self.print_debug(
            "[MaskRCNN-Help-3] no preprocess will be applied on model-transform, so that your input_img should be processed as model-zoo harness suggests!"
        )

    '''
        implement must after path_custom_dataset is valid in MaskRCNN_Tester_Basic
    '''

    def data_npz_path_init(self):
        self.path_input_top = self.path_default_MaskRCNN_dataset + "Superior_IMG_BackBone.npz"

        #also input of cascade_block::RPN2BboxPooler
        self.path_output_BackBone = self.path_default_MaskRCNN_dataset + "all_outputs_SwinTBackBone.npz"

        self.path_output_RPN2BboxPooler = self.path_default_MaskRCNN_dataset + "Reference_RPN2BboxPooler.npz"
        self.path_output_End2End = self.path_default_MaskRCNN_dataset + "Reference_End2End.npz"

    '''
        check if arch support such specific utest
    '''

    def check_support(self, case) -> bool:
        _, bm1684x_support = self.test_cases[case]
        if self.chip == "bm1684x" and bm1684x_support:
            return True
        return False

    '''
        test a single utest
    '''

    def test_single(self, case: str):
        np.random.seed(0)
        torch.manual_seed(7)
        MaskRCNN_IR_TESTER.ID = 0
        MaskRCNN_IR_TESTER.CURRENT_CASE = case
        self.print_debug("Test: {}".format(case))
        if case in self.test_cases:
            os.makedirs(case, exist_ok=True)
            os.chdir(case)
            func, _, = self.test_cases[case]
            func()
            self.print_debug("====== TEST {} Success ======".format(case))
        else:
            raise RuntimeError("case [{}] is not exist".format(case))

    '''
        get first torch.pt from model_def_multi_torch
    '''

    def gen_model_def(self) -> str:
        assert isinstance(self.model_def_multi_torch, list)
        assert self.model_def_multi_torch
        return self.model_def_multi_torch[0]

    '''
        get other but not first  torch.pt models from model_def_multi_torch,
        transform them into a long string, rather than list[str]
    '''

    def gen_model_extern(self) -> str:
        assert isinstance(self.model_def_multi_torch, list)
        if len(self.model_def_multi_torch) > 1:
            result = ",".join(self.model_def_multi_torch[1:])
            return result
        return None

    '''
        ensure no maskrcnn superior parameter is abundant for model_transform/deploy
    '''

    def info_packed_dict_checker(self, dict_test_MaskRCNN: dict):
        abundant_keys = [
            key for key in dict_test_MaskRCNN if key not in self.dict_MaskRCNN_packed_interface
        ]
        if len(abundant_keys) > 0:
            raise RuntimeError(
                "[MaskRCNN-Test-Error] user-given parameters: {} are illegal!".format(
                    abundant_keys))

    '''
        prepare cmd parameters for model_deploy
    '''

    def info_packed_str_cmd_params(self, dict_test_MaskRCNN: dict) -> dict:
        revised_dict = dict()
        for _, key in enumerate(dict_test_MaskRCNN.keys()):
            revised_dict[key] = ''.join(str(dict_test_MaskRCNN[key]).split())
        return revised_dict

    '''
        1)copy self.path_yaml from to  current utest file, ex. python/test/MaskRCNN_test_bm1684x/XX_Utest
        2)inject and revise default self.params_model_transform_MaskRCNN
        3)save revised  dict_content_yaml back to self.adjust_yaml
        4)test will read the modified self.adjust_yaml for utest
    '''

    def inject_YAML_config(self, dict_test_MaskRCNN: dict):
        assert isinstance(self.path_yaml,
                          str), "[MaskRCNN-Error]Only support path_yaml as pure string now!"
        assert os.path.exists(
            self.path_yaml), "[MaskRCNN-Test-Error] {} yaml path is not exist!".format(
                self.path_yaml)

        # now path is adjust by os.chdir
        # Step 1)
        self.adjust_yaml = str(pathlib.Path().resolve()) + "/MaskRCNN_Config.yaml"
        cmd = ["cp", self.path_yaml, self.adjust_yaml]
        _os_system(cmd)

        # Step 2)
        with open(self.path_yaml, 'r') as file:
            self.dict_content_yaml = yaml.unsafe_load(file)
            self.dict_content_yaml[self.params_model_transform_MaskRCNN] = dict()
            if "maskrcnn_io_map" in dict_test_MaskRCNN.keys():
                self.dict_content_yaml[self.params_model_transform_MaskRCNN][
                    'maskrcnn_io_map'] = dict_test_MaskRCNN["maskrcnn_io_map"]
            self.dict_content_yaml[self.params_model_transform_MaskRCNN][
                'maskrcnn_num_output'] = dict_test_MaskRCNN["maskrcnn_output_num"]
            self.dict_content_yaml[self.params_model_transform_MaskRCNN][
                'list_encoded_block'] = dict_test_MaskRCNN["maskrcnn_structure"]
            self.dict_content_yaml[self.params_model_transform_MaskRCNN][
                'list_name_PPLop'] = dict_test_MaskRCNN["maskrcnn_ppl_op"]
            self.dict_content_yaml[self.params_model_transform_MaskRCNN][
                'nums_PPLOp_Inputs'] = dict_test_MaskRCNN["numPPLOp_InWithoutWeight_MaskRCNN"]

        # Step 3)
        with open(self.adjust_yaml, 'w') as outfile:
            yaml.dump(self.dict_content_yaml, outfile)

    '''
      parameter: int/list/dict ===<info_packed_str_cmd_params>====>  str =====<inject_YAML_config>======> CONFIG_MaskRCNN.yaml ===<MaskRCNNConverter>===> int/list/dict
    '''

    def CMD_model_transformer(self, dict_test_MaskRCNN_native: dict, enable_maskrcnn: bool = True):
        dict_test_MaskRCNN = self.info_packed_str_cmd_params(dict_test_MaskRCNN_native)
        self.info_packed_dict_checker(dict_test_MaskRCNN)
        self.inject_YAML_config(dict_test_MaskRCNN)

        cmd = ["model_transform.py"]
        cmd.extend(["--model_def", dict_test_MaskRCNN["model_def"]])
        if "model_extern" in dict_test_MaskRCNN.keys():
            cmd.extend(["--model_extern", dict_test_MaskRCNN["model_extern"]])
        cmd.extend(["--model_name", dict_test_MaskRCNN["model_name"]])
        cmd.extend(["--input_shapes", dict_test_MaskRCNN["input_shapes"]])
        cmd.extend(["--mlir", "{}.mlir".format(dict_test_MaskRCNN["model_name"])])
        if (enable_maskrcnn):
            cmd.extend(["--enable_maskrcnn"])
        if (self.debug):
            cmd.extend(["--debug"])
        cmd.extend(["--path_yaml", self.adjust_yaml])
        _os_system(cmd)

    def CMD_model_deploy(self, dict_test_MaskRCNN_native: dict, enable_maskrcnn: bool = True):
        dict_test_MaskRCNN = self.info_packed_str_cmd_params(dict_test_MaskRCNN_native)
        self.info_packed_dict_checker(dict_test_MaskRCNN)

        cmd = ["model_deploy.py"]
        cmd.extend(["--mlir", "{}.mlir".format(dict_test_MaskRCNN["model_name"])])
        cmd.extend(["--chip", str(self.chip)])
        assert "." in self.model_file, "[MaskRCNN-Error] .bmodel not pure bmodel!"
        cmd.extend([
            "--model", "{}_{}_{}_num_core_{}{}".format(dict_test_MaskRCNN["model_name"], self.chip,
                                                       self.mode, self.num_core, self.model_file)
        ])
        cmd.extend(["--test_input", dict_test_MaskRCNN["test_input"]])
        cmd.extend(["--test_reference", dict_test_MaskRCNN["test_reference"]])
        if (enable_maskrcnn):
            cmd.extend(["--enable_maskrcnn"])
        if (self.debug):
            cmd.extend(["--debug"])
        _os_system(cmd)

    def test_MaskRCNN_Utest_SwintT_BackBone(self):
        cmd = ["model_transform.py"]
        model_name = "SwinT_BackBone"
        cmd.extend(["--model_name", model_name])
        cmd.extend(["--model_def", self.gen_model_def()])
        cmd.extend(["--input_shapes", "[1,3,800,1216]"])
        cmd.extend(["--mean", "0,0,0"])
        cmd.extend(["--scale", "1,1,1"])
        cmd.extend(["--keep_aspect_ratio"])
        cmd.extend(["--pixel_format", "rgb"])
        if (self.debug):
            cmd.extend(["--debug"])
        cmd.extend(["--test_input", self.path_input_top])
        cmd.extend(["--test_result", "{}_top_outputs.npz".format(model_name)])
        cmd.extend(["--mlir", "{}.mlir".format(model_name)])
        cmd.extend(["--tolerance", "0.99,0.85"])
        _os_system(cmd)

        cmd = ["model_deploy.py"]
        cmd.extend(["--mlir", "{}.mlir".format(model_name)])
        cmd.extend(["--quantize", "{}".format(self.mode)])
        cmd.extend(["--chip", "{}".format(self.chip)])
        cmd.extend(["--test_input", "{}_in_f32.npz".format(model_name)])
        cmd.extend(["--test_reference", "{}_top_outputs.npz".format(model_name)])
        cmd.extend([
            "--model", "{}_{}_{}_num_core_{}{}".format(model_name, self.chip, self.mode,
                                                       self.num_core, self.model_file)
        ])
        if (self.debug):
            cmd.extend(["--debug"])
            cmd.extend(["--compare_all"])
        cmd.extend(["--tolerance", "0.99,0.85"])
        _os_system(cmd)

    def test_MaskRCNN_Utest_RPNGetBboxes(self):
        dict_test_MaskRCNN = {}
        dict_test_MaskRCNN["model_def"] = "Model_Fake_Utest.pt"
        dict_test_MaskRCNN["model_name"] = "block_RPNGetBboxes"
        dict_test_MaskRCNN["input_shapes"] = [[1, 3, 200, 304], [1, 3, 100, 152], [1, 3, 50, 76],
                                              [1, 3, 25, 38], [1, 3, 13, 19], [1, 12, 200, 304],
                                              [1, 12, 100, 152], [1, 12, 50, 76], [1, 12, 25, 38],
                                              [1, 12, 13, 19], [1, 1, 4741, 4]]
        dict_test_MaskRCNN["maskrcnn_output_num"] = 1
        dict_test_MaskRCNN["maskrcnn_structure"] = 0
        dict_test_MaskRCNN["maskrcnn_ppl_op"] = "ppl::RPN_get_bboxes"
        dict_test_MaskRCNN["numPPLOp_InWithoutWeight_MaskRCNN"] = 11
        dict_test_MaskRCNN[
            "test_input"] = self.path_default_MaskRCNN_dataset + "input_mlir___get_bboxes_DQ_multi_batch__.npz"
        dict_test_MaskRCNN[
            "test_reference"] = self.path_default_MaskRCNN_dataset + "output_mlir___get_bboxes_DQ_multi_batch__.npz"
        # enable_maskrcnn
        self.CMD_model_transformer(dict_test_MaskRCNN)
        self.CMD_model_deploy(dict_test_MaskRCNN)

    def test_MaskRCNN_Utest_BboxPooler(self):
        dict_test_MaskRCNN = {}
        dict_test_MaskRCNN["model_def"] = "Model_Fake_Utest.pt"
        dict_test_MaskRCNN["model_name"] = "Utest_BboxPooler"
        dict_test_MaskRCNN["input_shapes"] = [[1, 256, 200, 304], [1, 256, 100, 152],
                                              [1, 256, 50, 76], [1, 256, 25, 38], [1, 250, 1, 5]]
        dict_test_MaskRCNN["maskrcnn_output_num"] = 2
        dict_test_MaskRCNN["maskrcnn_structure"] = 0
        dict_test_MaskRCNN["maskrcnn_ppl_op"] = "ppl::Bbox_Pooler"
        dict_test_MaskRCNN["numPPLOp_InWithoutWeight_MaskRCNN"] = 5
        dict_test_MaskRCNN[
            "test_input"] = self.path_default_MaskRCNN_dataset + "input_mlir___static_bbox_pooler_tester__.npz"
        dict_test_MaskRCNN[
            "test_reference"] = self.path_default_MaskRCNN_dataset + "output_mlir___static_bbox_pooler_tester__.npz"
        # enable_maskrcnn
        self.CMD_model_transformer(dict_test_MaskRCNN)
        self.CMD_model_deploy(dict_test_MaskRCNN)

    def test_MaskRCNN_Utest_GetBboxB(self):
        dict_test_MaskRCNN = {}
        dict_test_MaskRCNN["model_def"] = "Model_Fake_Utest.pt"
        dict_test_MaskRCNN["model_name"] = "Utest_GetBboxB"
        dict_test_MaskRCNN["input_shapes"] = [[1, 250, 1, 5], [1, 250, 1, 320], [1, 250, 1, 81],
                                              [1, 20000, 1, 4], [1, 1, 20000, 4]]
        dict_test_MaskRCNN["maskrcnn_output_num"] = 2
        dict_test_MaskRCNN["maskrcnn_structure"] = 0
        dict_test_MaskRCNN["maskrcnn_ppl_op"] = "ppl::get_bboxes_B"
        dict_test_MaskRCNN["numPPLOp_InWithoutWeight_MaskRCNN"] = 5
        dict_test_MaskRCNN[
            "test_input"] = self.path_default_MaskRCNN_dataset + "input_mlir___nodechip_MaskRCNNGetBboxB_global_tester__.npz"
        dict_test_MaskRCNN[
            "test_reference"] = self.path_default_MaskRCNN_dataset + "output_mlir___nodechip_MaskRCNNGetBboxB_global_tester__.npz"
        # enable_maskrcnn
        self.CMD_model_transformer(dict_test_MaskRCNN)
        self.CMD_model_deploy(dict_test_MaskRCNN)

    def test_MaskRCNN_Utest_MaskPooler(self):
        dict_test_MaskRCNN = {}
        dict_test_MaskRCNN["model_def"] = "Model_Fake_Utest.pt"
        dict_test_MaskRCNN["model_name"] = "Utest_MaskPooler"
        dict_test_MaskRCNN["input_shapes"] = [[1, 256, 200, 304], [1, 256, 100, 152],
                                              [1, 256, 50, 76], [1, 256, 25, 38], [1, 100, 1, 5],
                                              [1, 100, 1, 1], [1, 100, 1, 4]]
        dict_test_MaskRCNN["maskrcnn_output_num"] = 1
        dict_test_MaskRCNN["maskrcnn_structure"] = 0
        dict_test_MaskRCNN["maskrcnn_ppl_op"] = "ppl::Mask_Pooler"
        dict_test_MaskRCNN["numPPLOp_InWithoutWeight_MaskRCNN"] = 7
        dict_test_MaskRCNN[
            "test_input"] = self.path_default_MaskRCNN_dataset + "input_mlir___superior_mask_rcnn_mask_pooler__.npz"
        dict_test_MaskRCNN[
            "test_reference"] = self.path_default_MaskRCNN_dataset + "output_mlir___superior_mask_rcnn_mask_pooler__.npz"
        # enable_maskrcnn
        self.CMD_model_transformer(dict_test_MaskRCNN)
        self.CMD_model_deploy(dict_test_MaskRCNN)

    def test_MaskRCNN_Utest_BatchedNMS(self):
        assert 0, "delete now"
        dict_test_MaskRCNN = {}
        dict_test_MaskRCNN["model_def"] = "Model_Fake_Utest.pt"
        dict_test_MaskRCNN["model_name"] = "Utest_BachtedNMS"
        dict_test_MaskRCNN["input_shapes"] = [[1, 1, 4741, 4], [1, 1, 4741, 1], [1, 1, 4741, 1]]
        dict_test_MaskRCNN["maskrcnn_output_num"] = 1
        dict_test_MaskRCNN["maskrcnn_structure"] = 0
        dict_test_MaskRCNN["maskrcnn_ppl_op"] = "ppl::Batched_Nms"
        dict_test_MaskRCNN["numPPLOp_InWithoutWeight_MaskRCNN"] = 3
        dict_test_MaskRCNN[
            "test_input"] = self.path_default_MaskRCNN_dataset + "input_mlir___dq_batched_nms_single_batch__.npz"
        dict_test_MaskRCNN[
            "test_reference"] = self.path_default_MaskRCNN_dataset + "output_mlir___dq_batched_nms_single_batch__.npz"
        # enable_maskrcnn
        self.CMD_model_transformer(dict_test_MaskRCNN)
        self.CMD_model_deploy(dict_test_MaskRCNN)

    def test_MaskRCNN_RPN_to_BboxPooler(self):
        dict_test_MaskRCNN = {}
        dict_test_MaskRCNN["model_def"] = "Model_Fake_Utest.pt"
        dict_test_MaskRCNN["model_name"] = "block_RPN_to_BboxPooler"
        dict_test_MaskRCNN[
            "input_shapes"] = "[[1,256,200,304],[1,256,100,152],[1,256,50,76],[1,256,25,38],[1,3,200,304],[1,3,100,152],[1,3,50,76],[1,3,25,38],[1,3,13,19],[1,12,200,304],[1,12,100,152],[1,12,50,76],[1,12,25,38],[1,12,13,19],[1,1,4741,4]]"
        dict_test_MaskRCNN["maskrcnn_output_num"] = 2  #Same as last bock: BboxPooler
        dict_test_MaskRCNN["maskrcnn_structure"] = "0,0"
        dict_test_MaskRCNN["maskrcnn_ppl_op"] = "ppl::RPN_get_bboxes,ppl::Bbox_Pooler"
        dict_test_MaskRCNN["numPPLOp_InWithoutWeight_MaskRCNN"] = "11,5"

        dict_test_MaskRCNN[
            "maskrcnn_io_map"] = "{(0,0):(-1,4),(0,1):(-1,5),(0,2):(-1,6),(0,3):(-1,7),(0,4):(-1,8),(0,5):(-1,9),(0,6):(-1,10),(0,7):(-1,11),(0,8):(-1,12),(0,9):(-1,13),(0,10):(-1,14),(1,0):(-1,0),(1,1):(-1,1),(1,2):(-1,2),(1,3):(-1,3),(1,4):(0,0),(-2,0):(1,0),(-2,1):(1,1)}"

        #Input Gen from Backbone output: all_outputs_SwinTBackBone.npz. In this example, it'snot same as input of first block:  RPNGetBboxes
        mode_input_generator = "RPN2BBOXPOOLER"
        Superior_MaskRCNN_InputParser = MaskRCNN_InputPreprocessor(
            path_yaml=self.path_yaml,
            path_input_image=self.path_output_BackBone,
            path_preprocessed_npz=None,
            basic_max_shape_inverse=self.basic_max_shape_inverse,
            basic_scalar_factor=self.basic_scalar_factor,
            debug=self.debug,
            mode_input_generator=mode_input_generator)

        #Input auto save as MaskRCNN_InputPreprocessor.path_save_preprocessed
        dict_test_MaskRCNN["test_input"] = Superior_MaskRCNN_InputParser.path_save_preprocessed
        #Output usually different from the reference of last bock: BboxPooler
        dict_test_MaskRCNN["test_reference"] = self.path_output_RPN2BboxPooler
        # enable_maskrcnn
        self.CMD_model_transformer(dict_test_MaskRCNN)
        self.CMD_model_deploy(dict_test_MaskRCNN)

    def test_MaskRCNN_End2End(self):
        dict_test_MaskRCNN = {}
        assert isinstance(self.model_def_multi_torch, list), self.model_def_multi_torch
        assert self.model_def_multi_torch
        dict_test_MaskRCNN["model_def"] = self.gen_model_def()
        dict_test_MaskRCNN["model_extern"] = self.gen_model_extern()
        dict_test_MaskRCNN["model_name"] = "SwinT_MaskRCNN_End2End"
        dict_test_MaskRCNN[
            "input_shapes"] = "[[1,3,800,1216],[1,1,4741,4],[1,1,20000,4],[1,1,20000,4],[1,1,100,4]]"
        dict_test_MaskRCNN["maskrcnn_output_num"] = 3
        dict_test_MaskRCNN["maskrcnn_structure"] = "1,0,0,1,0,0,1"
        dict_test_MaskRCNN[
            "maskrcnn_ppl_op"] = "ppl::RPN_get_bboxes,ppl::Bbox_Pooler,ppl::get_bboxes_B,ppl::Mask_Pooler"
        dict_test_MaskRCNN["numPPLOp_InWithoutWeight_MaskRCNN"] = "11,5,5,7"

        dict_test_MaskRCNN[
            "maskrcnn_io_map"] = "{(0,0):(-1,0),(1,0):(0,5),(1,1):(0,6),(1,2):(0,7),(1,3):(0,8),(1,4):(0,9),(1,5):(0,10),(1,6):(0,11),(1,7):(0,12),(1,8):(0,13),(1,9):(0,14),(1,10):(-1,1),(2,0):(0,0),(2,1):(0,1),(2,2):(0,2),(2,3):(0,3),(2,4):(1,0),(3,0):(2,0),(4,0):(2,1),(4,1):(3,1),(4,2):(3,0),(4,3):(-1,2),(4,4):(-1,3),(5,0):(0,0),(5,1):(0,1),(5,2):(0,2),(5,3):(0,3),(5,4):(4,0),(5,5):(4,1),(5,6):(-1,4),(6,0):(5,0),(-2,0):(4,0),(-2,1):(4,1),(-2,2):(6,0)}"

        #Input Gen from Input: a single output
        mode_input_generator = "Complete"
        Superior_MaskRCNN_InputParser = MaskRCNN_InputPreprocessor(
            path_yaml=self.path_yaml,
            path_input_image=self.path_input_top,
            path_preprocessed_npz=None,
            basic_max_shape_inverse=self.basic_max_shape_inverse,
            basic_scalar_factor=self.basic_scalar_factor,
            debug=self.debug,
            mode_input_generator=mode_input_generator)

        # Input auto save as MaskRCNN_InputPreprocessor.path_save_preprocessed
        dict_test_MaskRCNN["test_input"] = Superior_MaskRCNN_InputParser.path_save_preprocessed
        # Output of complete MaskRCNN
        dict_test_MaskRCNN["test_reference"] = self.path_output_End2End
        # enable_maskrcnn
        self.CMD_model_transformer(dict_test_MaskRCNN)
        self.CMD_model_deploy(dict_test_MaskRCNN)


def test_one_case_in_all(tester: MaskRCNN_IR_TESTER, case, error_cases, success_cases):
    t = Timer()
    try:
        tester.test_single(case)
    except:
        error_cases.append("{}:{}s".format(case, int(t.elapsed_time())))
        traceback.print_exc()
        return
    success_cases.append("{}:{}s".format(case, int(t.elapsed_time())))


def test_all(tester: MaskRCNN_IR_TESTER):
    error_cases = []
    success_cases = []
    for case in tester.test_cases:
        if tester.check_support(case):
            test_one_case_in_all(tester, case, error_cases, success_cases)
    print("Success: {}".format(success_cases))
    print("Failure: {}".format(error_cases))
    if error_cases:
        print("====== test_MaskRCNN.py --chip {} TEST Failed ======".format(tester.chip))
        # exit(1)
    else:
        print("====== test_MaskRCNN.py --chip {} TEST Success ======".format(tester.chip))
    clean_kmp_files()
    return error_cases


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # yapf: disable
    parser.add_argument("--chip", default="bm1684x", type=str,
                        choices=['bm1684x'], help="chip platform name")
    parser.add_argument("--case", default="all", type=str, help="test one case, if all, then test all cases")
    parser.add_argument("--mode", default="f32", type=str, choices=['f32'],
                        help="chip platform name")
    parser.add_argument("--num_core", default=1, type=int, choices=[1], help='The numer of TPU cores used for parallel computation')
    parser.add_argument("--debug", action="store_true", help='keep middle file if debug')
    parser.add_argument("--show_all", action="store_true", help='show all cases')
    parser.add_argument("--debug_cmd", default="", type=str, help="debug_cmd")
    parser.add_argument("--path_yaml", type=str, default=None, help="one YAML recording MaskRCNN parameters")

    #used for Backbone or complete
    parser.add_argument("--model_def_multi_torch", type=str2list, default=None,  help="torch model definition file.")
    parser.add_argument("--path_custom_dataset", type=str, default=None,  help="custom_dataset, if defaut nnmodels is impossibly reached")


    # yapf: enable
    args = parser.parse_args()
    tester = MaskRCNN_IR_TESTER(model_def_multi_torch=args.model_def_multi_torch,
                                chip=args.chip,
                                mode=args.mode,
                                debug=args.debug,
                                num_core=args.num_core,
                                debug_cmd=args.debug_cmd,
                                path_yaml=args.path_yaml,
                                path_custom_dataset=args.path_custom_dataset)

    if args.show_all:
        print("====== Show All Cases ============")
        for case in tester.test_cases:
            print(case)
        exit(0)
    dir = "MaskRCNN_test_{}".format(args.chip)
    os.makedirs(dir, exist_ok=True)
    os.chdir(dir)
    if args.case == "" or args.case.lower() == "all":
        test_all(tester)
    else:
        tester.test_single(args.case)
    if args.debug == False:
        file_clean()
'''
#Utest Single PPLOp
python3 test_MaskRCNN.py --case MaskRCNN_Utest_RPNGetBboxes --debug
python3 test_MaskRCNN.py --case MaskRCNN_Utest_BboxPooler   --debug
python3 test_MaskRCNN.py --case MaskRCNN_Utest_GetBboxB     --debug
python3 test_MaskRCNN.py --case MaskRCNN_Utest_MaskPooler   --debug

#Only BackBone
python3 test_MaskRCNN.py \
    --case MaskRCNN_Utest_SwinTBackBone --debug \
    --model_def_multi_torch /workspace/tpu-mlir/python/test/test_case_tpu_mlir_maskrcnn/test_case/mask_rcnn_swin_T_3_part1_trace_model_part1_Backbone_and_RPN_weight.pt

#Cascade blocks
python3 test_MaskRCNN.py --case MaskRCNN_RPN_to_BboxPooler --debug

#Complete SwinTMaskRCNN End2End
export PATH_PT_MASKRCNN=/workspace/tpu-mlir/python/test/test_case_tpu_mlir_maskrcnn/test_case
python3 test_MaskRCNN.py \
    --case MaskRCNN_End2End --debug \
    --model_def_multi_torch $PATH_PT_MASKRCNN/mask_rcnn_swin_T_3_part1_trace_model_part1_Backbone_and_RPN_weight.pt,$PATH_PT_MASKRCNN/mr_swinT_bbox_head_4.pt,$PATH_PT_MASKRCNN/mr_swinT_mask_head_4.pt
'''
