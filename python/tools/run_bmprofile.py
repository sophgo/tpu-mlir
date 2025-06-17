#!/usr/bin/env python3
# ==============================================================================
#
# Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
#
# TPU-MLIR is licensed under the 2-Clause BSD License except for the
# third-party components.
#
# ==============================================================================

from typing import List, Dict
import os
import shutil
import subprocess
import re
import time
from tools.bmodel_dis import BModelCMDIter
from debugger import disassembler as dis
import fire
from datetime import datetime


def BModel2Bin(bmodel_file):
    import math

    class FName:
        core_id = 0
        subnet_id = 0
        gid = 0
        length = 0
        suffix = ""

        def __str__(self):
            return os.path.splitext(bmodel_file)[0] + f"{self.suffix}.{self.core_id}"

    fname = FName()
    bmodel = dis.BModel(bmodel_file)
    core_num = bmodel.core_num
    for subnet in BModelCMDIter(bmodel):
        fname.core_id = 0
        fname.subnet_id = subnet.id
        for fname.gid, cmds in enumerate(subnet.cmd_group):
            fname.length, fname.suffix = cmds.tiu_num, ".BD"
            with open(str(fname), "wb") as f:
                f.write(bytes(cmds.tiu_cmd))
            fname.length, fname.suffix = cmds.dma_num, ".GDMA"
            with open(str(fname), "wb") as f:
                f.write(bytes(cmds.dma_cmd))
        for fname.core_id, _cmds in enumerate(subnet.core_commands):
            for fname.gid, cmds in enumerate(_cmds.gdma_tiu_commands):
                fname.length, fname.suffix = cmds.tiu_num, ".BD"
                with open(str(fname), "wb") as f:
                    f.write(bytes(cmds.tiu_cmd))
                fname.length, fname.suffix = cmds.dma_num, ".GDMA"
                with open(str(fname), "wb") as f:
                    f.write(bytes(cmds.dma_cmd))
            for fname.gid, cmds in enumerate(_cmds.sdma_commands):
                # cmds contains system-end.
                fname.length, fname.suffix = math.ceil(len(cmds) / 96), ".SDMA"
                with open(str(fname), "wb") as f:
                    f.write(bytes(cmds))
            for fname.gid, cmds in enumerate(_cmds.hau_commands):
                fname.length, fname.suffix = math.ceil(len(cmds) / 80), ".HAU"
                with open(str(fname), "wb") as f:
                    f.write(bytes(cmds))
            for fname.gid, cmds in enumerate(_cmds.cdma_commands):
                fname.length, fname.suffix = math.ceil(len(cmds) / 120), ".HAU"
                with open(str(fname), "wb") as f:
                    f.write(bytes(cmds))
    return bmodel


def log_bmodel_shape(bmodel_path):
    bmodel = dis.BModel(bmodel_path)
    input_tensors = bmodel.net[0].parameter[0].input_tensor
    shape_list = []
    for i in input_tensors:
        shape = i.shape[0]
        shape_list.append([int(x) for x in shape])
    new_shape_str = ":".join(["x".join([str(x) for x in s]) for s in shape_list])
    print("bmodel_shape:", new_shape_str)


class ProgramRunner:

    def __init__(self):
        pass

    def find_binary_cmds_files(self, folder_path):
        result_dict = {}
        patterns = {
            "TIU": ".*.BD(.[0-9])?",
            "GDMA": ".*.GDMA(.[0-9])?",
            "SDMA": ".*.SDMA(.[0-9])?",
            "HAU": ".*.HAU(.[0-9])?",
            "CDMA": ".*.CDMA(.[0-9])?",
            "Reg": "GDMA_reg.*.txt",
            "Sreg": "SDMA_reg.*.txt",
        }
        for root, dirs, files in os.walk(folder_path):
            tempdict = {}
            max_number = -1
            for file_name in files:
                for key, pattern in patterns.items():
                    if re.search(pattern, file_name):
                        file_path = os.path.abspath(os.path.join(root, file_name))
                        tempdict[key] = tempdict.get(key, []) + [file_path]
                        # Extract and update the max number
                        number_match = re.search(r"\.([0-9])\.", file_name)
                        if number_match:
                            number = int(number_match.group(1))
                            max_number = max(max_number, number)

            if tempdict and (len(tempdict) > 1 or "Reg" not in tempdict):
                result_dict.update(tempdict)
                result_dict["core_num"] = len(tempdict["GDMA"])
        return result_dict

    def find_output_folders(self, result_path):
        output_dict = {}
        for root, dirs, files in os.walk(result_path):
            if "output" in dirs:
                folder_name = root[len(result_path) + 1:]
                output_path = os.path.abspath(os.path.join(root, "output"))
                output_dict[folder_name] = output_path
        return output_dict

    def generate_command_single(self, keys, file_dict):
        command_parts = []
        mapping = {
            "tiu": f'--tiuBuf "{file_dict["tiu"]}"',
            "dma": f'--dmaBuf "{file_dict["dma"]}"',
            "sample": f'--dram_preload {file_dict.get("sample", "")}',
            "reg": f'--gdmaReg {file_dict.get("reg", "")}',
        }

        for key in keys:
            if key in mapping:
                command_parts.append(mapping[key])

        command_string = " ".join(command_parts)
        return command_string

    def generate_sg2260_command(self, file_items: Dict[str, List[str]]):

        command_parts = []

        mapping = {
            "TIU": "--tiuBuf {file_name}",
            "GDMA": "--dmaBuf {file_name}",
            "SDMA": "--sdmaBuf {file_name}",
            "CDMA": "--cdmaBuf {file_name}",
            "HAU": "--hauBuf {file_name}",
            "Reg": "--gdmaReg {file_name}",
            "Sreg": "--sdmaReg {file_name}",
        }

        for key, value in file_items.items():
            if key in mapping:
                left, *_ = value[0].rsplit(".", maxsplit=1)
                command_parts.append(mapping[key].format(file_name=left))
                # if len(value) == 1:
                # else:
                #     command_parts.append(mapping[key].format(file_name=value[0]))

        command_string = " ".join(command_parts)
        return command_string

    def generate_command_single(self, keys, file_dict):
        command_parts = []
        mapping = {
            "tiu": f'--tiuBuf "{file_dict["tiu"]}"',
            "dma": f'--dmaBuf "{file_dict["dma"]}"',
            "sample": f'--dram_preload {file_dict.get("sample", "")}',
            "reg": f'--gdmaReg {file_dict.get("reg", "")}',
        }

        for key in keys:
            if key in mapping:
                command_parts.append(mapping[key])

        command_string = " ".join(command_parts)
        return command_string

    def execute_program(self, file_dict, chip: str, core_num, script_path, style, gen_web, configs):
        """
        Execute program in each model folder and then generate corresponding files
        The generated files will then move to the subfolder/output folder
        The SummaryInfo.csv file will incorporate data info for each model
        """
        main_directory = os.getcwd()
        # folder_name, file_dict = item[0], item[1]

        profile_raw_path = os.path.join(profile_path)
        os.makedirs(profile_raw_path, exist_ok=True)
        os.chdir(profile_raw_path)
        chip = chip.upper()
        if configs:
            print("configs:", configs)
            destination = os.path.join(profile_raw_path, "configs.json")
            shutil.copy2(configs, destination)
        if chip == "BM1684X":
            exe_path = "tpu_cmodel"
        elif chip == "BM1688" or chip == "CV186X":
            exe_path = "tpuTwoCore"
        elif chip == "BM1690":
            exe_path = "tpuEightCore"
        else:
            raise NotImplementedError(chip)

        if not shutil.which(exe_path):
            raise FileNotFoundError(f"{exe_path} not found")

        command_string = self.generate_sg2260_command(file_dict)
        command = f"{exe_path} {command_string}"
        print("About to execute command:", command)

        try:
            subprocess.run(command, shell=True, check=True, env=os.environ)
            print("Command executed successfully")
        except subprocess.CalledProcessError as e:
            print(f"Command execution failed: {e}")
        except Exception as e:
            print(f"Failure: {e}")

        os.chdir(main_directory)

    def parallel_processing(self, bmodel_binary_dir, chip, style, gen_web, configs):
        script_path = dict()
        cmd_files = self.find_binary_cmds_files(bmodel_binary_dir)

        if cmd_files:
            core_num = cmd_files.get("core_num", 8)
            self.execute_program(
                file_dict=cmd_files,
                chip=chip,
                core_num=core_num,
                script_path=script_path,
                style=style,
                gen_web=gen_web,
                configs=configs,
            )
        else:
            raise FileNotFoundError()


def find_summaryinfo(folder_path):
    summaryinfo = []
    for root, dirs, files in os.walk(folder_path):
        if "output" in root:  # 进入output文件夹
            temp = {}
            for file_name in files:
                if "CycleInfo" in file_name:
                    temp["csvfinal"] = os.path.abspath(os.path.join(root, file_name))
                elif "profile" in file_name:
                    temp["profile"] = os.path.abspath(os.path.join(root, file_name))
                elif "simulatorTotalCycle" in file_name:
                    temp["simulator"] = os.path.abspath(os.path.join(root, file_name))
            if "profile" not in temp.keys():
                temp["profile"] = os.path.abspath(os.path.join(root, "compiler_profile_0.txt"))
            if len(temp) == 3:
                summaryinfo.append(temp["csvfinal"])
                summaryinfo.append(temp["profile"])
                summaryinfo.append(temp["simulator"])
    return summaryinfo


def generate_result_folders(source_directory, result_directory):
    for root, dirs, files in os.walk(source_directory):
        relative_path = os.path.relpath(root, source_directory)
        result_path = os.path.join(result_directory, relative_path)
        os.makedirs(result_path, exist_ok=True)
        if "output" not in dirs:
            continue
        output_path = os.path.join(root, "output")
        result_output_path = os.path.join(result_path, "output")
        shutil.move(output_path, result_output_path)


def get_rootdir_path(path):
    return os.path.dirname(os.path.dirname(os.path.abspath(path)))


def recreate(dir):
    if os.path.exists(dir):
        assert os.path.isdir(dir)
        shutil.rmtree(dir)
    os.makedirs(dir, exist_ok=True)


def get_base_name(bmodel: str):

    dirname = os.path.dirname(os.path.abspath(bmodel))
    basename = os.path.basename(bmodel)

    if basename == "compilation.bmodel":
        profile_base = dirname.rstrip("/")
    else:
        context_dir = os.path.join(dirname, os.path.splitext(basename)[0])
        profile_base = context_dir.rstrip("/")
    return os.path.join(os.path.dirname(profile_base), "diagnostic")


def simulation(bmodel: str):
    """
    Here update the data by "git pull"
    :arg1: the path containing executable ie. tpuEightCore;
    :arg2: the folder containing instructions. It can have multiple subdirectories inside.
    :arg3: the chip arg. Case insensitive.
    :arg4(optional): the choice of setting style for excels (default is true)
    :arg5(optional): the choice of generate perfai.web (default is true)
    :arg6(optional): the choice of setting specified configs.json
    e.g. python3 tool/AutoGenRegInfo.py ./out/bin/ TPUPerfBenchZoo/SG2260
    e.g. python3 tool/AutoGenRegInfo.py ./out/bin/ TPUPerfBenchZoo/SG2260 --toolonly TPUPerfBenchZoo/result_SG2260 --style 1 --gen_web 1
    """
    nowdir = os.getcwd()
    assert ("PERFAI_SIMULATION_ROOT" in os.environ), "should assign PERFAI_SIMULATION_ROOT"

    bmodel_abs_path = os.path.abspath(bmodel)
    bmodel_base_name = os.path.basename(bmodel_abs_path)
    base_root = os.path.dirname(bmodel_abs_path)
    bmodel_ext0 = os.path.splitext(bmodel_base_name)[0]
    log_bmodel_shape(bmodel_abs_path)
    bmodel_binary_dir = os.path.join(base_root, f"{bmodel_ext0}.binary")

    profile_raw_name = os.path.join(get_base_name(bmodel), f"perfai_profile_raw")
    shutil.rmtree(profile_raw_name, ignore_errors=True)

    profile_raw_dir = os.path.join(base_root, profile_raw_name)

    bmodel_ = dis.BModel(bmodel_abs_path)
    chip = bmodel_.chip

    def bmodel2binary():
        recreate(bmodel_binary_dir)
        shutil.copy2(bmodel_abs_path, bmodel_binary_dir)
        BModel2Bin(os.path.join(bmodel_binary_dir, bmodel_base_name))
        return bmodel_binary_dir

    def create_reg():
        config_path = ""
        runner = ProgramRunner()
        if chip == "BM1690":
            CHIP_ARCH = "sg2260"
        elif chip == "BM1684X":
            CHIP_ARCH = "bm1684x"
        elif chip == "CV186X" or chip == "BM1688":
            CHIP_ARCH = "A2"
        else:
            raise NotImplementedError(chip)

        os.environ["CHIP_ARCH"] = CHIP_ARCH
        os.environ["PATH"] = os.pathsep.join([
            os.environ["PATH"],
            os.path.join(os.environ["PERFAI_SIMULATION_ROOT"], CHIP_ARCH),
        ])

        global profile_path
        profile_path = profile_raw_dir
        recreate(profile_raw_dir)
        runner.parallel_processing(bmodel_binary_dir,
                                   chip=chip,
                                   style=0,
                                   gen_web=0,
                                   configs=config_path)

    start = time.time()
    bmodel_binary_dir = bmodel2binary()

    create_reg()
    end = time.time()
    passed = end - start
    print(f"Total spent time: {passed:.2f} seconds")

    return profile_raw_dir


def bmprofile(path: str):
    name = str(datetime.now()).replace(":", "_").replace(" ", "_")
    bmodel = dis.BModel(path)
    chip = bmodel.chip
    remote = os.environ.get(f"REMOTE_{chip.upper()}")
    if remote is None:
        raise Exception(f"env REMOTE_{chip.upper()} not found")

    ret, log = subprocess.getstatusoutput(f"ssh {remote} mkdir {name}")
    ret, log = subprocess.getstatusoutput(f"scp {path} {remote}:{name}")

    basename = os.path.basename(path)

    ret, log = subprocess.getstatusoutput(
        f"""ssh {remote} 'export BMRUNTIME_ENABLE_PROFILE=1 ; cd {name} ; /opt/sophon/libsophon-current/bin/bmrt_test --bmodel {basename}' """
    )
    if ret != 0:
        raise RuntimeError(f"bmrt_test failed: {log}")

    profile_raw = os.path.join(get_base_name(path), f"bm_profile_raw")

    try:
        shutil.rmtree(profile_raw)
    except Exception:
        pass

    os.makedirs(os.path.dirname(profile_raw), exist_ok=True)
    ret, log = subprocess.getstatusoutput(f"scp -r {remote}:{name}/bmprofile_data-1 {profile_raw}")
    return profile_raw


if __name__ == "__main__":
    entry = {
        "simulation": simulation,
        "bmprofile": bmprofile,
    }

    fire.Fire(entry)
