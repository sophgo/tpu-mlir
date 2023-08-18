#!/usr/bin/env python3
# Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
#
# TPU-MLIR is licensed under the 2-Clause BSD License except for the
# third-party components.
#
# ==============================================================================

from tools.npz_tool import npz_compare
from utils.preprocess import supported_customization_format
from utils.mlir_shell import _os_system
from chip import *
import configparser

from tools.model_transform import *
from utils.mlir_shell import *
import os
import threading
import queue


class MODEL_RUN(object):

    def __init__(self,
                 model_name: str,
                 chip: str = "bm1684x",
                 mode: str = "all",
                 dyn_mode: bool = False,
                 merge_weight: bool = False,
                 fuse_preprocess: bool = False,
                 customization_format: str = "",
                 aligned_input: bool = False,
                 save_log: bool = False,
                 disable_thread: bool = True,
                 debug:bool = False,
                 num_core: int = 1):
        self.model_name = model_name
        self.chip = chip
        self.mode = mode
        self.dyn_mode = dyn_mode
        self.fuse_pre = fuse_preprocess
        self.customization_format = customization_format
        self.aligned_input = aligned_input
        self.merge_weight = merge_weight
        self.save_log = save_log
        self.disable_thread = disable_thread
        self.debug = debug
        self.model_type = chip_support[self.chip][-1]
        self.command = f"run_model.py {model_name} --chip {chip} --mode {mode}"
        self.num_core = num_core

        config = configparser.ConfigParser(inline_comment_prefixes=('#', ))
        config.read(os.path.expandvars(f"$REGRESSION_PATH/config/{self.model_name}.ini"))
        # save all content in model config file as dict
        self.ini_content = dict(config.items("DEFAULT"))
        # replace env vars with true values
        for key in self.ini_content:
            self.ini_content[key] = os.path.expandvars(self.ini_content[key])

        if not os.path.exists(self.ini_content["model_path"]):
            if "model_path2" not in self.ini_content or not os.path.exists(
                    self.ini_content["model_path2"]):
                raise RuntimeError("{} not exist".format(self.ini_content["model_path"]))
            self.ini_content["model_path"] = self.ini_content["model_path2"]

        self.do_cali = not self.ini_content["model_path"].endswith(".tflite")
        self.arch = chip
        if chip.startswith("cv18") and chip != "cv186x":
            self.arch = "cv18xx"
        elif chip == "bm1686" or chip == "cv186x":
            self.arch = "bm1684x"
        self.tolerance = {
            "f32": config.get(self.arch, "f32_tolerance", fallback="0.99,0.99"),
            "f16": config.get(self.arch, "f16_tolerance", fallback="0.95,0.85"),
            "bf16": config.get(self.arch, "bf16_tolerance", fallback="0.95,0.80"),
            "int8_sym": config.get(self.arch, "int8_sym_tolerance", fallback="0.8,0.5"),
            "int8_asym": config.get(self.arch, "int8_asym_tolerance", fallback="0.8,0.5"),
            "int4_sym": config.get(self.arch, "int4_sym_tolerance", fallback="0.8,0.5"),
        }
        # set quant_modes according to argument and config files
        self.quant_modes = {
            "f32": 0,
            "f16": 0,
            "bf16": 0,
            "int8_sym": 0,
            "int8_asym": 0,
            "int4_sym": 0,
        }
        self.do_test = "test_input" in self.ini_content
        if self.ini_content["model_path"].endswith(".tflite"):
            self.quant_modes["int8_sym"] = 1
        else:
            if self.mode != "all" and self.mode != "basic":
                self.quant_modes[self.mode] = 1
            else:
                self.quant_modes["int8_sym"] = 1
                if self.mode == "all":
                    self.quant_modes["f16"] = 1
                    self.quant_modes["bf16"] = 1
                    self.quant_modes["f32"] = 1
        for idx, quant_mode in enumerate(self.quant_modes.keys()):
            if f"do_{quant_mode}" in self.ini_content:
                self.quant_modes[quant_mode] &= int(self.ini_content[f"do_{quant_mode}"])
            # check chip support from chip.py
            if quant_mode == mode:
                assert (chip_support[self.chip][idx]
                        and "Current chip doesn't support this quant mode")
            self.quant_modes[quant_mode] &= chip_support[self.chip][idx]

        self.do_dynamic = self.dyn_mode and ("do_dynamic" in self.ini_content and int(
            self.ini_content["do_dynamic"])) and chip_support[self.chip][-2]

    def run_model_transform(self, model_name: str, dynamic: bool = False):
        '''transform from origin model to top mlir'''
        cmd = ["model_transform.py"]
        # add required arguments
        top_result = f"{model_name}_top_outputs.npz"
        # static test_reference and input_npz won't be used in model_deploy
        if not model_name.endswith("_static"):
            self.ini_content["test_reference"] = top_result
            self.ini_content["input_npz"] = f"{model_name}_in_f32.npz"
        cmd.extend([
            f"--model_name {model_name}", f"--mlir {model_name}.mlir",
            "--model_def {}".format(self.ini_content["model_path"])
        ])
        if "model_data" in self.ini_content:
            cmd += ["--model_data {}".format(self.ini_content["model_data"])]
        if self.do_test:
            cmd.extend([
                "--test_input {}".format(self.ini_content["test_input"]),
                f"--test_result {top_result}"
            ])
        if self.debug:
            cmd += ["--debug"]
        # add preprocess infor
        if dynamic:
            cmd += ["--input_shapes {}".format(self.ini_content["dynamic_shapes"])]
        elif "input_shapes" in self.ini_content:
            cmd += ["--input_shapes {}".format(self.ini_content["input_shapes"])]
        if "input_types" in self.ini_content:
            cmd += ["--input_types {}".format(self.ini_content["input_types"])]
        if "resize_dims" in self.ini_content:
            cmd += ["--resize_dims {}".format(self.ini_content["resize_dims"])]
        if "keep_aspect_ratio" in self.ini_content and int(self.ini_content["keep_aspect_ratio"]):
            cmd += ["--keep_aspect_ratio"]
        if "mean" in self.ini_content:
            cmd += ["--mean {}".format(self.ini_content["mean"])]
        if "scale" in self.ini_content:
            cmd += ["--scale {}".format(self.ini_content["scale"])]
        if "pixel_format" in self.ini_content:
            cmd += ["--pixel_format {}".format(self.ini_content["pixel_format"])]
        if "channel_format" in self.ini_content:
            cmd += ["--channel_format {}".format(self.ini_content["channel_format"])]
        if "pad_value" in self.ini_content:
            cmd += ["--pad_value {}".format(self.ini_content["pad_value"])]
        if "pad_type" in self.ini_content:
            cmd += ["--pad_type {}".format(self.ini_content["pad_type"])]

        # add others
        if "output_names" in self.ini_content:
            cmd += ["--output_names {}".format(self.ini_content["output_names"])]
        if "excepts" in self.ini_content:
            cmd += ["--excepts {}".format(self.ini_content["excepts"])]
        _os_system(cmd, self.save_log)

    def make_calibration_table(self):
        '''generate calibration when there is no existing one'''
        if "specified_cali_table" in self.ini_content.keys():
            self.cali_table = self.ini_content["specified_cali_table"]
        else:
            self.cali_table = os.path.expandvars(
                f"$REGRESSION_PATH/cali_tables/{self.model_name}_cali_table")
        if os.path.exists(self.cali_table):
            return
        if "dataset" not in self.ini_content:
            raise RuntimeError("[!Error]: dataset not set for calibration")
        cmd = ["run_calibration.py"]
        cmd.extend([
            f"{self.model_name}.mlir", "--dataset {}".format(self.ini_content["dataset"]),
            f"-o {self.cali_table}"
        ])
        if "tune_num" in self.ini_content:
            t_num = self.ini_content["tune_num"]
            cmd.extend([f"--tune_num {t_num}"])
        if "input_num" in self.ini_content:
            i_num = self.ini_content["input_num"]
            cmd.extend([f"--input_num {i_num}"])
        if "debug_cmd" in self.ini_content:
            d_cmd = self.ini_content["debug_cmd"]
            cmd.extend([f"--debug_cmd {d_cmd}"])
        else:
            cmd.extend(["--input_num 100"])
        _os_system(cmd, self.save_log)

    def int4_tmp_test(self):
        '''tmp test script for int4 sym mode, no bmodel generated for now'''

        # generate tpu mlir
        tpu_mlir = f"{self.model_name}_bm1686_tpu_int4_sym.mlir"
        if "use_quantize_table" in self.ini_content and int(
                self.ini_content["use_quantize_table"]):
            qtable=self.cali_table.replace("_cali_table","_qtable")
            cmd = [
                "tpuc-opt", f"{self.model_name}.mlir", "--chip-assign=\"chip=bm1686\"",
                "--chip-top-optimize",
                f"--import-calibration-table=\"file={self.cali_table} asymmetric=false\"",
                f"--convert-top-to-tpu=\"mode=INT4 qtable={qtable} asymmetric=false\"", "--canonicalize",
                f"-o {tpu_mlir}"
            ]
        else:
            cmd = [
                "tpuc-opt", f"{self.model_name}.mlir", "--chip-assign=\"chip=bm1686\"",
                "--chip-top-optimize",
                f"--import-calibration-table=\"file={self.cali_table} asymmetric=false\"",
                "--convert-top-to-tpu=\"mode=INT4 asymmetric=false\"", "--canonicalize",
                f"-o {tpu_mlir}"
            ]
        _os_system(cmd, self.save_log)

        # inference and compare
        output_npz = tpu_mlir.replace(".mlir", "_outputs.npz")
        cmd = [
            "model_runner.py", "--input {}_in_f32.npz".format(self.model_name),
            f"--model {tpu_mlir}", f"--output {output_npz}"
        ]
        _os_system(cmd, self.save_log)
        cmd = ["npz_tool.py", "compare", output_npz, self.ini_content["test_reference"], "-v"]
        if "int4_sym_tolerance" in self.ini_content:
            cmd += "--tolerance {}".format(self.ini_content["int4_sym_tolerance"]),
        if "excepts" in self.ini_content:
            cmd += ["--excepts {}".format(self.ini_content["excepts"])]

        _os_system(cmd, self.save_log)

    def test_input_copy(self, quant_mode):
        test_input = self.ini_content["test_input"] if self.fuse_pre else self.ini_content[
            "input_npz"]
        new_test_input = ""
        if self.fuse_pre:
            new_test_input = test_input.replace(".jpg", f"_for_{quant_mode}.jpg").split("/")[-1]
        else:
            new_test_input = test_input.replace(".npz", f"_for_{quant_mode}.npz")
        cmd = ["cp", test_input, new_test_input]
        _os_system(cmd, self.save_log)
        return new_test_input

    def run_model_deploy(self,
                         quant_mode: str,
                         model_name: str,
                         dynamic: bool = False,
                         test: bool = True,
                         do_sample: bool = False):
        '''top mlir -> bmodel/ cvimodel'''
        # int4_sym mode currently in test
        to_test = self.do_test and test
        if to_test:
            new_test_input = self.test_input_copy(quant_mode)

        if quant_mode == "int4_sym":
            self.int4_tmp_test()
            return

        cmd = ["model_deploy.py"]

        # add according to arguments
        model_file = f"{model_name}_{self.chip}_{quant_mode}"
        if self.fuse_pre:
            cmd += ["--fuse_preprocess"]
            model_file += "_fuse_preprocess"
        if self.aligned_input:
            cmd += ["--aligned_input"]
            model_file += "_aligned_input"
        if self.customization_format:
            cmd += [f"--customization {self.customization_format}"]
        if self.merge_weight:
            cmd += ["--merge_weight"]
            model_file += "_merge_weight"
        if self.debug:
            cmd += ["--debug"]
        if self.num_core != 1:
            cmd += [f"--num_core {self.num_core}"]

        # add for int8 mode
        if (quant_mode.startswith("int8") or quant_mode.startswith("int4")):
            if self.do_cali:
                cmd += [f"--calibration_table {self.cali_table}"]
                if "use_quantize_table" in self.ini_content and int(
                        self.ini_content["use_quantize_table"]):
                    qtable = self.cali_table.replace("_cali_table", "_qtable")
                    cmd += [f"--quantize_table {qtable}"]
            if self.ini_content["model_path"].endswith(".tflite"):
                cmd += ["--asymmetric"]
            else:
                if not ("quant_input" in self.ini_content
                        and int(self.ini_content["quant_input"]) == 0):
                    cmd += ["--quant_input"]
                if not ("quant_output" in self.ini_content
                        and int(self.ini_content["quant_output"]) == 0):
                    cmd += ["--quant_output"] if self.model_type == "bmodel" else [""]

        # add for dynamic mode
        if dynamic:
            cmd += ["--dynamic"]

        # add the rest
        model_file += f".{self.model_type}"
        cmd.extend([
            "--mlir {}.mlir".format(model_name if not dynamic else self.model_name),
            f"--chip {self.chip}",
            "--compare_all",
            f"--model {model_file}",
            "--quantize {}".format(quant_mode.replace("_sym", "").replace("_asym", "").upper()),
            "--tolerance {}".format(self.tolerance[quant_mode]),
        ])
        if to_test:
            cmd.extend([
                "--test_input {}".format(new_test_input),
                "--test_reference {}".format(self.ini_content["test_reference"])
            ])
        if "excepts" in self.ini_content:
            cmd += ["--excepts {}".format(self.ini_content["excepts"])]

        _os_system(cmd, self.save_log)

        if to_test:
            os.system(f"rm {new_test_input}")

        # only run sample for f32 and int8_sym mode
        if do_sample and (quant_mode == "f32" or quant_mode == "int8_sym"):
            output_file = self.model_name + f"_{quant_mode}.jpg"
            self.run_sample(model_file, self.ini_content["test_input"], output_file)

    def run_dynamic(self, quant_mode: str):
        '''do dynamic regression
            1. do static model_transform (with dynamic_shapes)
            2. do static model deploy (based on the top mlir generated in step 1. no test input compare)
            3. do dynamic model deploy (based on the origin top mlir)
            4. compare bmodel inference result of static and dynamic
        '''

        static_model_name = self.model_name + "_static"
        dyn_model_name = self.model_name + "_dynamic"
        self.run_model_transform(static_model_name, dynamic=True)

        out_suffix = f"_out_{quant_mode}.npz"
        static_out = static_model_name + out_suffix
        dyn_out = dyn_model_name + out_suffix

        # static model with dynamic_shapes doesn't do result compare
        static_model_file = self.run_model_deploy(quant_mode, static_model_name, test=False)
        dyn_model_file = self.run_model_deploy(quant_mode, dyn_model_name, dynamic=True)

        cmd = [
            "model_runner.py", f"--input {static_model_name}_in_f32.npz",
            f"--model {static_model_file}", f"--output {static_out}"
        ]
        _os_system(cmd, self.save_log)
        cmd[2], cmd[3] = f"--model {dyn_model_file}", f"--output {dyn_out}"
        _os_system(cmd, self.save_log)
        cmd = ["npz_tool.py", "compare", static_out, dyn_out, "-vv"]
        _os_system(cmd, self.save_log)

    def run_sample(self, model_def: str, test_input: str, output: str, model_data: str = ""):
        '''run samples under tpu-mlir/python/test/'''

        cmd = [
            self.ini_content["app"], f"--model {model_def}", f"--input {test_input}",
            f"--output {output}"
        ]
        if model_data:
            cmd += [f"--model_data {model_data}"]

        _os_system(cmd, self.save_log)

    def run_model_deploy_wrapper(self, quant_mode, model_name, do_sample, result_queue):
        try:
            self.run_model_deploy(quant_mode, model_name, False, True, do_sample)
            result_queue.put((quant_mode, True, None))
        except Exception as e:
            result_queue.put((quant_mode, False, e))

    def run_full(self):
        '''run full process: model_transform, model_deploy, samples and dynamic mode'''
        try:
            do_sample = "app" in self.ini_content and not self.chip.startswith("cv")
            if do_sample:
                # origin model
                self.run_sample(
                    self.ini_content["model_path"], self.ini_content["test_input"],
                    self.model_name + "_origin.jpg",
                    self.ini_content["model_data"] if "model_data" in self.ini_content else "")

            self.run_model_transform(self.model_name)
            if (self.quant_modes["int4_sym"] or self.quant_modes["int8_sym"]
                    or self.quant_modes["int8_asym"]) and self.do_cali:
                self.make_calibration_table()
            if self.disable_thread:
                for quant_mode, support in self.quant_modes.items():
                    if support:
                        result_queue = queue.Queue()
                        self.run_model_deploy_wrapper(quant_mode, self.model_name, do_sample,
                                                      result_queue)
                        _, success, error = result_queue.get()
                        if not success:
                            raise error
            else:
                result_queue = queue.Queue()
                threads = []
                for quant_mode in self.quant_modes.keys():
                    if self.quant_modes[quant_mode]:
                        t = threading.Thread(target=self.run_model_deploy_wrapper,
                                             args=(quant_mode, self.model_name, do_sample,
                                                   result_queue))
                        t.start()
                        threads.append(t)

                for t in threads:
                    t.join()

                while not result_queue.empty():
                    _, success, error = result_queue.get()
                    if not success:
                        raise error

            # currently only do f32 dynamic mode
            if self.do_dynamic and self.quant_modes["f32"]:
                self.run_dynamic("f32")
            print("Success: {}".format(self.command))
            return 0
        except RuntimeError as e:
            print(repr(e))
            print("Failed: {}".format(self.command))
            return 1
        except:
            print("Unknown errors")
            print("Failed: {}".format(self.command))
            return 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # yapf: disable
    parser.add_argument('model_name', metavar='model_name', help='model name')
    parser.add_argument("--out_dir", default="", type=str, help="out directory")
    parser.add_argument("--chip", default="bm1684x", type=str.lower, help="chip platform name")
    parser.add_argument("--mode", default="all", type=str.lower,
                        choices=['all', 'basic', 'f32', 'f16', 'bf16', 'int8_sym', 'int8_asym', 'int4_sym'],
                        help="quantize mode, 'all' runs all modes except int4, 'baisc' runs f16 and int8 sym only")
    parser.add_argument("--dyn_mode", default='store_true', help="dynamic mode")
    parser.add_argument("--merge_weight", action="store_true",
                        help="merge weights into one weight binary with previous generated cvimodel")
    # fuse preprocess
    parser.add_argument("--fuse_preprocess", action='store_true',
                        help="add tpu preprocesses (mean/scale/channel_swap) in the front of model")
    parser.add_argument("--customization_format", default='', type=str.upper,
                        choices=supported_customization_format,
                        help="pixel format of input frame to the model")
    parser.add_argument("--aligned_input", action='store_true',
                        help='if the input frame is width/channel aligned')
    parser.add_argument("--save_log", action="store_true", help='if true, save the log to file')
    parser.add_argument("--disable_thread", action="store_true", help='do test without multi thread')
    parser.add_argument("--debug", action="store_true", help='keep middle file if debug')
    parser.add_argument("--num_core", default=1, type=int,
                        help="The number of TPU cores used for parallel computation.")
    # yapf: enable
    args = parser.parse_args()
    out_dir = f"$REGRESSION_PATH/regression_out/{args.model_name}_{args.chip}_num_core_{args.num_core}" if args.out_dir == "" else args.out_dir
    dir = os.path.expandvars(out_dir)
    os.makedirs(dir, exist_ok=True)
    os.chdir(dir)
    runner = MODEL_RUN(args.model_name, args.chip, args.mode, args.dyn_mode, args.merge_weight,
                       args.fuse_preprocess, args.customization_format, args.aligned_input,
                       args.save_log, args.disable_thread, args.debug, args.num_core)
    runner.run_full()
