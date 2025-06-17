import time
import pandas as pd
import sys
from pathlib import Path
import os
import logging
from argparse import Namespace
from utils.mlir_shell import _os_system_log, getstatusoutput_v2
from pydantic import BaseModel
from utils.cache_tool import CommandRecorder
from typing import Optional, Callable, List, Any
from utils.log_setting import setup_logger
import json
import math
import re
import shutil
import textwrap
import functools
from pydantic import Field

_DEPTH = 0

BASE_DIAGNOSITIC_NAME = f"diagnostic"


class PropertyCheck(BaseModel):
    k: str
    v: Any
    failed_action: str = "raise"


class SkipTask(Exception):
    pass


class TaskDep:

    def __init__(self) -> None:
        self.verbose = 0
        self.ignore = False
        self.in_check = False
        self._DEPTH = 0
        self.gen_file_map = {}

        self.task_map = {}

    def check(
        that,
        *,
        depend_tasks: List[Callable] = None,
        depend_files: List[str] = None,
        gen_files: List[str] = None,
        properties: List[PropertyCheck] = None,
        force: bool = False,
    ):

        def wrapper(func: Callable):
            if gen_files:
                for file in gen_files:
                    that.gen_file_map.setdefault(file, []).append(func.__name__)
            that.task_map[func.__name__] = func

            def inner(self, target_file: str, *args, **kwargs):
                recorder = CommandRecorder(target_file, read=True)
                target = RefFile(**recorder.dic)
                logger.info(textwrap.indent(f"check [{func.__name__}] context", "  " * that._DEPTH))

                should_run = False

                that.ignore = that.ignore or kwargs.get("ignore", False)
                that.verbose = that.verbose or kwargs.get("verbose", None)

                if kwargs.get("verbose"):
                    kwargs.pop("verbose")
                    logger.setLevel(logging.DEBUG)

                def check_depend_tasks():
                    if not depend_tasks:
                        return False
                    logger.info(textwrap.indent("run dependent tasks", "  " * that._DEPTH))
                    ret = False
                    that._DEPTH += 1
                    for task in depend_tasks:
                        if isinstance(task, str):
                            task = that.task_map[task]

                        try:
                            task(self, target_file, *args, **kwargs)
                            ret = True
                        except SkipTask:
                            pass
                    that._DEPTH -= 1
                    return ret

                recorder.refresh()

                def check_files():
                    if depend_files:
                        for file in depend_files:
                            cont = getattr(target.files, file)
                            if cont is None or not os.path.exists(cont.path):
                                logger.error(f"{file} not found")
                                if file in that.gen_file_map:
                                    for func in that.gen_file_map[file]:
                                        logger.error(
                                            f" try to run {func.replace('task_', '')} command to generate {file}"
                                        )
                                raise FileNotFoundError(f"{file} not found")

                            if _file_modified(cont) and not that.ignore:
                                raise ValueError(
                                    f"{file} is modified, {cont.last_modify}, got {os.path.getmtime(cont.path)}"
                                )

                def check_properties():
                    if not properties:
                        return False
                    for check in properties:
                        if getattr(target.properties, check.k) != check.v:
                            if check.failed_action == "raise":
                                raise ValueError(
                                    f"{check.k} is modified, {getattr(target.properties, check.k)}, got {check.v}"
                                )
                            else:
                                return True
                    return False

                def check_gen_files():
                    if not gen_files:
                        return False
                    for gen_file in gen_files:
                        if not _all_file_no_change(getattr(target.files, gen_file)):
                            return True
                    return False

                check_files()

                should_run = check_depend_tasks()
                if not should_run:
                    should_run = check_properties()
                if not should_run:
                    should_run = check_gen_files()

                if that.ignore or force:
                    should_run = True

                if depend_tasks is None and gen_files is None and properties is None:
                    should_run = True

                if should_run:
                    if force:
                        logger.info(
                            textwrap.indent(f"-> force run [{func.__name__}]", "  " * that._DEPTH))
                    else:
                        logger.info(textwrap.indent(f"-> run [{func.__name__}]",
                                                    "  " * that._DEPTH))
                    that._DEPTH += 1
                    try:
                        that.in_check = False
                        logger.debug(f"close skip raise before {func.__name__}")
                        ret = func(self, target_file, *args)
                        that.in_check = True
                        logger.debug(f"open skip raise after {func.__name__}")
                        logger.info(
                            textwrap.indent(f" * run {func.__name__} success", "  " * that._DEPTH))
                    except SkipTask:
                        ret = None
                        logger.debug(f"skip some tasks in {func.__name__}")

                    that._DEPTH -= 1
                    return ret
                else:
                    logger.info(textwrap.indent(f"skip [{func.__name__}]", "  " * that._DEPTH))
                    if that._DEPTH > 0 and that.in_check:
                        raise SkipTask()

            setattr(inner, "__doc__", func.__doc__)
            return inner

        return wrapper


check = TaskDep().check


class FileFlag(BaseModel):
    path: str
    last_modify: float


class DebugFile(BaseModel):
    # input
    mlir_input: Optional[FileFlag] = None

    # ir
    origin_model: Optional[FileFlag] = None
    origin_mlir: Optional[FileFlag] = None
    top_mlir: Optional[FileFlag] = None
    tpu_mlir: Optional[FileFlag] = None
    tpu_opt_mlir: Optional[FileFlag] = None
    final_mlir: Optional[FileFlag] = None
    bmodel: Optional[FileFlag] = None

    # final directory
    context_dir: Optional[FileFlag] = None

    # middle file
    tensor_location: Optional[FileFlag] = None
    layer_group_cache: Optional[FileFlag] = None
    bmodel_failed_summary: Optional[FileFlag] = None

    bmodel_failed_tensor: Optional[FileFlag] = None
    ## bmodel_checker.py COMB_ALL output
    bmodel_inference: Optional[FileFlag] = None
    ## npz_tool.py
    history_compare_file: Optional[FileFlag] = None
    ## layer-group >> lg.log
    lg_log: Optional[FileFlag] = None
    ## perfai simulation output
    simulation_dir: Optional[FileFlag] = None
    ## runtime output
    bmprofile_dir: Optional[FileFlag] = None
    ## perfai output
    profile_csv: Optional[FileFlag] = None
    profile_web: Optional[FileFlag] = None
    ## mlir2graph output
    tpu_lowered_svg: Optional[FileFlag] = None
    tpu_addressed_svg: Optional[FileFlag] = None

    # output
    top_output: Optional[FileFlag] = None
    tpu_output: Optional[FileFlag] = None
    bmodel_output: Optional[FileFlag] = None

    # mlir2onnx output
    top_onnx: Optional[FileFlag] = None
    tpu_onnx: Optional[FileFlag] = None
    tpu_opt_onnx: Optional[FileFlag] = None


class DebugProperty(BaseModel):
    chip: Optional[str] = None
    deploy_pwd: Optional[str] = None
    compare_all: Optional[bool] = None
    cache_mode: Optional[str] = None
    prefix: Optional[str] = None


class DebugCommand(BaseModel):
    tpu_opt: Optional[str] = None
    final: Optional[str] = None
    bmodel: Optional[str] = None
    deploy_cmd: Optional[str] = None


class RefFile(BaseModel):
    version: str
    commands: DebugCommand = Field(default_factory=DebugCommand)
    files: DebugFile = Field(default_factory=DebugFile)
    properties: DebugProperty = Field(default_factory=DebugProperty)


class DebugData(BaseModel):
    cmd: str
    ref_files: str
    good: Optional[str]
    target: DebugFile
    reference: Optional[DebugFile]


class DebugItArgs(Namespace):
    cmd: str
    ref_files: str
    good: str


def _get_bmodel_outputs(context_dir, tpu_reference: str):
    return {
        "file_dep": [context_dir, tpu_reference],
        "targets": [],
        "actions": [
            f"bmodel_checker.py {context_dir} {tpu_reference} --no_interactive -C 'check dump_mode COMB_ALL'"
        ],
    }


def _fno2cmd_ids(tl_df):
    """
    {file_line: Dict[core_id, Tuple[List[(tiu_0, tiu1)], List[(dma_0, dma1)]]]}
    """
    fno2cmd_ids = {}
    for filn, subdf in tl_df.groupby("file-line"):
        core_cmd_ids = {}
        for _, core_id, (tiu0, dma0), (tiu1, dma1) in zip(
                list(subdf.index),
                list(subdf["core_id"]),
                list(subdf["tiu_dma_id(before)"]),
                list(subdf["tiu_dma_id(after)"]),
        ):
            tiu, dma = core_cmd_ids.setdefault(core_id, ([], []))

            if tiu1 - tiu0 > 0:
                tiu.append([tiu0, tiu1])
            if dma1 - dma0 > 0:
                dma.append([dma0, dma1])

        fno2cmd_ids[filn] = core_cmd_ids
    return fno2cmd_ids


def _load(tensor_location: str):
    tl = json.loads(Path(tensor_location).read_text())
    location = json.loads(Path(tensor_location).read_text())
    df = pd.DataFrame.from_records(location)
    return df


def _parse_group_cmd_id_range(groups, tl_df):
    lino_group_id = {}
    min_id = {}
    max_id = {}

    def drop_nan(val, default):
        if math.isnan(val):
            val = default
        return val

    core_ids = sorted(set(tl_df["core_id"]))
    for k, v in groups.items():

        for vv in v:
            lino_group_id[vv] = k

            def find_core(core_id):
                fdf = tl_df[(tl_df["core_id"] == core_id) & (tl_df["file-line"] == vv)]

                tiu = fdf["tiu_dma_id(before)"].apply(lambda x: x[0]).min()
                dma = fdf["tiu_dma_id(before)"].apply(lambda x: x[1]).min()
                dic = min_id.setdefault(k, {})
                min_tiu, min_dma = dic.get(core_id, (10**9, 10**9))

                dic[core_id] = (
                    min(drop_nan(tiu, 10**9), min_tiu),
                    min(drop_nan(dma, 10**9), min_dma),
                )

                tiu = fdf["tiu_dma_id(after)"].apply(lambda x: x[0]).max()
                dma = fdf["tiu_dma_id(after)"].apply(lambda x: x[1]).max()
                dic = max_id.setdefault(k, {})
                max_tiu, max_dma = dic.get(core_id, (0, 0))
                dic[core_id] = (
                    max(drop_nan(tiu, 0), max_tiu),
                    max(drop_nan(dma, 0), max_dma),
                )

            for _core_id in core_ids:
                find_core(_core_id)
    return lino_group_id, min_id, max_id


def get_bmodel_outputs(args: DebugData):
    yield _get_bmodel_outputs(args.target.context_dir, args.target.tpu_output)
    if args.good:
        yield _get_bmodel_outputs(args.reference.context_dir, args.reference.tpu_output)


def _parse_groups(final_mlir):
    """
    get a dict of parse result,
    "groups": group_idx -> [lineno]
    "op_num": group_idx -> [lineno]
    """
    lines = Path(final_mlir).read_text().split("\n")

    group = None
    core_group = None
    group_parallel = None

    groups = {}
    group_line = []
    group_with_line = {}

    for line_no, line in enumerate(lines, start=1):
        if "tpu.GroupParallel" in line:
            group_parallel = True
            continue
        if "#tpu<core_pattern Common>" in line:
            group_parallel = False
            continue

        if "tpu.Group" in line:
            group = True
            continue
        if "tpu.CoreParallel" in line:
            core_group = True
            continue
        if "tpu.Yield" in line:
            group = False

        if line.startswith("      }) "):
            group = False
            core_group = False
        if group or core_group:
            if "tpu.Split" not in line:
                group_line.append(line_no)
                if "Yeild" not in line and "Join" not in line:
                    if "Load" not in line and "Store" not in line:
                        if core_group:
                            if len(groups) not in group_with_line:
                                group_with_line.setdefault(len(groups), []).append(line)
                        else:
                            group_with_line.setdefault(len(groups), []).append(line)

        elif group_parallel:
            if "tpu" in line and "tpu.Yield" not in line:
                groups[len(groups)] = [line_no]
        elif len(group_line) > 0:
            groups[len(groups)] = group_line.copy()
            group_line.clear()
        elif all([i not in line for i in ["tpu.Weight", "top.None", "top.Input", "tpu.Buffer"]]):
            if "tpu." in line and line.startswith("      %"):
                # print(line)
                group_with_line.setdefault(len(groups), []).append(line)
                groups[len(groups)] = [line_no]
            pass

    return {
        "groups": groups,
        "group_with_line": group_with_line,
    }


logger: logging.Logger = setup_logger("debugit", replace_root=True)


def _file_modified(file: FileFlag):
    if os.path.exists(file.path):
        if os.path.isfile(file.path):
            return os.path.getmtime(file.path) != file.last_modify
        else:
            return (os.path.getmtime(os.path.join(file.path, ".modify")) != file.last_modify)
    return False


def _all_file_no_change(*files: FileFlag):
    """file exists and no change comp"""
    if len(files) == 0:
        return True

    for file in files:
        if file is None:
            return False
        if not os.path.exists(file.path):
            return False

        if (os.path.getmtime(
                file.path if os.path.isfile(file.path) else os.path.join(file.path, ".modify"))
                != file.last_modify):

            return False

    return True


def load_ref_file(file: str):
    recorder = CommandRecorder(file, read=True)
    return RefFile(**recorder.dic)


entry = {}


class DebugBase:

    def __init__(self):
        for func in dir(self):
            if func.startswith("task_"):
                entry[func.replace("task_", "")] = getattr(self, func)

    @check()
    def task_final2graph(self, target_file: str):
        """
        draw graph of final.mlir

        try include any useful info
        """
        target_recorder = CommandRecorder(target_file, read=True)
        target = RefFile(**target_recorder.dic)

        cmd = [
            "mlir2graph.py",
            "--mlir",
            target.files.final_mlir.path,
            "--ref_file",
            os.path.abspath(target_file),
            "--mlir_order",
            "--force_order",
            "2",
        ]
        if _all_file_no_change(target.files.bmodel_failed_tensor):
            cmd.extend([
                "--bmodel_checker_data",
                target.files.bmodel_failed_tensor.path,
            ])

        if _all_file_no_change(target.files.history_compare_file):
            cmd.extend(["--failed_key_list", target.files.history_compare_file.path])

        if _all_file_no_change(target.files.final_mlir):
            getstatusoutput_v2(
                " ".join(cmd),
                shell=True,
                cwd=os.path.dirname(target.files.final_mlir.path),
                print_output=False,
            )
        else:
            logger.error(f"final_mlir {target.files.final_mlir.path} not exists")

    @check()
    def task_mlirs(self, target_file: str):
        import shutil

        recorder = CommandRecorder(target_file, read=True)
        target = RefFile(**recorder.dic)

        top_mlir = target.files.top_mlir
        tpu_mlir = target.files.tpu_mlir
        tpu_opt_mlir = target.files.tpu_opt_mlir
        final_mlir = target.files.final_mlir

        if top_mlir is None:
            raise RuntimeError("top_mlir not found")

        base_dir = os.path.join(os.path.dirname(top_mlir.path), BASE_DIAGNOSITIC_NAME, "mlirs")
        os.makedirs(base_dir, exist_ok=True)
        for file in [top_mlir, tpu_mlir, tpu_opt_mlir, final_mlir]:
            if file and os.path.exists(file.path):
                shutil.copy2(file.path, os.path.join(base_dir, os.path.basename(file.path)))
        logger.info(f"mlir files are saved in {base_dir}")

    @check()
    def task_mlir2onnx(self, target_file: str):
        from utils.mlir_shell import mlir2onnx

        recorder = CommandRecorder(target_file, read=True)
        target = RefFile(**recorder.dic)

        top_mlir = target.files.top_mlir
        tpu_mlir = target.files.tpu_mlir
        tpu_opt_mlir = target.files.tpu_opt_mlir

        if tpu_mlir is None:
            raise RuntimeError("tpu_mlir not found")

        base_dir = os.path.join(os.path.dirname(tpu_mlir.path), BASE_DIAGNOSITIC_NAME, "vis_onnx")
        os.makedirs(base_dir, exist_ok=True)

        if top_mlir:
            mlir2onnx(top_mlir.path, os.path.join(base_dir, f"top.onnx"))

        if tpu_mlir:
            mlir2onnx(tpu_mlir.path, os.path.join(base_dir, f"tpu.onnx"))

        if tpu_opt_mlir:
            mlir2onnx(tpu_opt_mlir.path, os.path.join(base_dir, f"tpu_opt.onnx"))

    @check()
    def task_tpu2graph(self, target_file: str):
        """draw graph of tpu.mlir"""
        target_recorder = CommandRecorder(target_file, read=True)
        target = RefFile(**target_recorder.dic)
        cmd = [
            "mlir2graph.py",
            "--mlir",
            target.files.tpu_mlir.path,
            "--ref_file",
            os.path.abspath(target_file),
            "--mlir_order",
            "--force_order",
            "2",
        ]
        if _all_file_no_change(target.files.layer_group_cache):
            cmd.extend(["--layer_group_cache", target.files.layer_group_cache.path])

        if _all_file_no_change(target.files.bmodel_failed_tensor):
            cmd.extend(["--bmodel_checker_data", target.files.bmodel_failed_tensor.path])

        if _all_file_no_change(target.files.history_compare_file):
            cmd.extend(["--failed_key_list", target.files.history_compare_file.path])

        if _all_file_no_change(target.files.tpu_mlir):
            getstatusoutput_v2(
                " ".join(cmd),
                cwd=os.path.dirname(target.files.tpu_mlir.path),
                shell=True,
                check=True,
                print_output=False,
            )

        else:
            logger.error(f"tpu_mlir {target.files.tpu_mlir.path} not exists")

    # def task_diff(self, target_file: str, reference_file: str):
    #     """display differences between target and reference"""
    #     target_recorder = CommandRecorder(target_file)
    #     reference_recorder = CommandRecorder(reference_file)
    #     target = RefFile(**target_recorder.dic.get("files", {}))
    #     reference = RefFile(**reference_recorder.dic.get("files", {}))

    #     _diff_mlir(target.files.tpu_mlir, reference.files.tpu_mlir)
    #     _diff_mlir(target.files.tpu_mlir, reference.files.tpu_mlir)
    #     _diff_mlir(target.files.final_mlir, reference.files.final_mlir)
    #     _diff_tensor_location(
    #         target.files.tensor_location, reference.files.tensor_location
    #     )
    @check()
    def task_redeploy(self, target_file: str):
        """re-run deploy command"""
        recorder = CommandRecorder(target_file, read=True)
        target = RefFile(**recorder.dic)

        cwd = None

        if target.properties.deploy_pwd:
            cwd = os.path.dirname(target.properties.deploy_pwd)
        if cwd is None and target.files.tpu_mlir:
            cwd = os.path.dirname(target.files.tpu_mlir.path)

        if cwd is None:
            raise RuntimeError("can't find deploy command cwd")

        getstatusoutput_v2(
            target.commands.deploy_cmd.replace("--debug", "") + " --debug",
            shell=True,
            check=True,
            print_output=True,
            cwd=cwd,
        )

    @check(depend_files=["final_mlir", "tensor_location"])
    def task_final_expand(
        self,
        target_file: str,
        verbose: int = 0,
    ):
        """
        get an expanded mlir file for debug

        verbose:
        1 for keep attr
        2 for use bmodel_failed_tensor
        """
        import numpy as np
        import re

        match_loceq = re.compile("(#loc[0-9]+) = loc\((.*)\)")
        recorder = CommandRecorder(target_file, read=True)
        target = RefFile(**recorder.dic)
        mlir = target.files.final_mlir.path
        tl = target.files.tensor_location.path
        content = Path(mlir).read_text()
        keep_attr = False
        use_bmodel_checker_data = False

        if verbose >= 1:
            keep_attr = True
        if verbose >= 2:
            use_bmodel_checker_data = True

        # locrefs = match_locref.findall(content)

        failed_key = set()
        if use_bmodel_checker_data and _all_file_no_change(target.files.bmodel_failed_tensor):
            bmodel_checker_data = target.files.bmodel_failed_tensor.path
            if bmodel_checker_data.endswith("npz"):
                failed_arr = np.load(bmodel_checker_data)
                failed_key.update(
                    {k.split("_asm_")[0]
                     for k in list(failed_arr.files) if "actual" in k})
            else:
                failed_key.update(
                    [i.strip() for i in Path(bmodel_checker_data).read_text().splitlines()])

        failed_key = [f'"{i}"' for i in failed_key]
        loceq = match_loceq.findall(content)
        for locref, locname in loceq:
            if locname in failed_key:
                content = content.replace(f"loc({locref})", f"loc({locref})({locname})(failed)")
            else:
                content = content.replace(f"loc({locref})", f"loc({locref})({locname})")

        lines = content.splitlines()

        parse_ret = _parse_groups(mlir)
        groups = parse_ret["groups"]

        df = _load(tl)

        fno2cmd_ids = _fno2cmd_ids(df)
        content = match_loceq.sub("", content).rstrip()

        new_lines = []
        if not keep_attr:
            for line in lines:
                left = line.find("{")
                right = line.rfind("}")
                if left > 0 and right > 0 and left < right:
                    line = line[:left] + line[right + 1:]
                new_lines.append(line)
            lines = new_lines

        for group_no, linos in groups.items():
            if len(linos) == 1:
                tgt_no = linos[0] - 1
            else:
                tgt_no = linos[0] - 2
            while "tpu.Split" in lines[tgt_no]:
                tgt_no -= 1

            lines[tgt_no] = re.sub("^(\s*)%", f"\\1group({group_no}) :{tgt_no}%", lines[tgt_no])

        for rlino in sorted(fno2cmd_ids, reverse=True):

            multi_core_cmd_ids = fno2cmd_ids[rlino]
            newline = re.sub("([^,(]\s*)%", f"\\1:{rlino}%", lines[rlino - 1])
            # newline = lines[rlino - 1].replace(" %", f" :{rlino}%")
            row = [newline]
            for core_id in sorted(multi_core_cmd_ids):
                tiu, dma = multi_core_cmd_ids[core_id]
                tiu = ", ".join([":".join(map(str, i)) for i in tiu])
                dma = ", ".join([":".join(map(str, i)) for i in dma])
                if tiu:
                    row.append(f"              tiu{core_id}: {tiu}")
                if dma:
                    row.append(f"              dma{core_id}: {dma}")

            lines[rlino - 1] = "\n".join(row)

        content = "\n".join(lines)
        Path(f"{mlir}.loc.mlir").write_text(content)
        logger.info(f"write to {mlir}.loc.mlir")


class DebugMetric(DebugBase):

    @check(depend_files=["context_dir", "tpu_output"], force=True)
    def task_tdb(self, target_file: str, reference_file: str = None):
        """run tdb directly"""
        recorder = CommandRecorder(target_file, read=True)
        target = RefFile(**recorder.dic)
        from tools.tdb import get_tdb

        ref_data = target.files.tpu_output.path

        if reference_file:
            ref_recorder = CommandRecorder(reference_file, read=True)
            reference = RefFile(**ref_recorder.dic)
            if _all_file_no_change(reference.files.bmodel_inference):
                ref_data = reference.files.bmodel_inference.path
                logger.info("use reference bmodel inference result for compare")

        command_args = [target.files.context_dir.path, "--ref_data", ref_data]
        logger.debug(f"run tdb.py {' '.join(command_args)}")
        tdb = get_tdb(command_args)
        tdb.cmdloop()

    @check(
        depend_files=["mlir_input", "tpu_mlir"],
        properties=[PropertyCheck(k="compare_all", v=True, failed_action="rerun")],
    )
    def task_tpu_output(self, target_file: str):
        """
        get all tpu.mlir inference result (dump_all)
        """
        recorder = CommandRecorder(target_file, read=True)
        target = RefFile(**recorder.dic)

        if _all_file_no_change(target.files.mlir_input, target.files.tpu_mlir,
                               target.files.tpu_output):
            getstatusoutput_v2(
                " ".join([
                    "model_runner.py",
                    "--input",
                    target.files.mlir_input.path,
                    "--model",
                    target.files.tpu_mlir.path,
                    "--output",
                    target.files.tpu_output.path,
                    "--dump_all_tensors",
                ]),
                shell=True,
                check=True,
                cwd=os.path.dirname(target.files.tpu_mlir.path),
            )
            recorder.add_file(tpu_output=target.files.tpu_output.path)
            recorder.add_property(compare_all=True)
            recorder.dump()

    @check()
    def task_bmodel_output(self, target_file: str):
        """
        get bmodel inference result
        """
        target_recorder = CommandRecorder(target_file, read=True)
        target = RefFile(**target_recorder.dic)
        if target.files.context_dir and target.files.tpu_output:
            ret, output = getstatusoutput_v2(
                f"bmodel_checker.py {target.files.context_dir.path} {target.files.tpu_output.path} --no_interactive -C 'check dump_mode COMB_ALL' --quiet",
                shell=True,
                cwd=target.files.context_dir.path,
                print_output=False,
            )
            if ret != 0:
                raise RuntimeError(f"bmodel_checker failed: {output}")

    @check()
    def task_bmodel_checker_cache(self, target_file: str):
        """prepare soc infer cache data"""
        target_recorder = CommandRecorder(target_file, read=True)
        target = RefFile(**target_recorder.dic)

        if target.files.context_dir and target.files.tpu_output:
            ret, output = getstatusoutput_v2(
                f"bmodel_checker.py {target.files.context_dir.path} {target.files.tpu_output.path} --no_interactive --cache_mode generate --quiet",
                shell=True,
                cwd=target.files.context_dir.path,
                print_output=False,
            )
            if ret != 0:
                raise RuntimeError(f"bmodel_checker failed: {output}")

    @check(depend_tasks=[task_tpu_output])
    def task_bmodel_checker(self, target_file: str):
        target_recorder = CommandRecorder(target_file, read=True)
        target = RefFile(**target_recorder.dic)
        if target.files.context_dir is None:
            raise RuntimeError("context_dir not found")

        if target.files.tpu_output is None:
            raise RuntimeError(
                "tpu_output not found, you should run `redeploy` command to generate debug files")

        from tools.bmodel_checker import main
        from debugger.tdb_support import CACHE_MODE_VERSION
        import platform

        argv = [
            target.files.context_dir.path,
            target.files.tpu_output.path,
        ]

        if platform.machine() != "x86_64":
            if target.properties.cache_mode is None:
                raise RuntimeError(
                    "cache data not found, you should try to run bmodel_checker_cache command to generate cache data"
                )
            if CACHE_MODE_VERSION != target.properties.cache_mode:
                raise RuntimeError(
                    f"cache_mode mismatch: {CACHE_MODE_VERSION} != {target.properties.cache_mode}, you should try to re-run bmodel_checker_cache command to generate cache data"
                )
            argv.extend([
                "--cache_mode",
                "offline",
            ])
        main(argv)

    @check()
    def task_npz_bmodel_compare(self, target_file: str, reference_file: str):
        """compare bmodel inference npz result of target and reference"""
        target_recorder = CommandRecorder(target_file, read=True)
        target = RefFile(**target_recorder.dic)
        reference = RefFile(**CommandRecorder(reference_file, read=True)._dic)

        save_file = os.path.join(os.path.dirname(target.files.bmodel_inference.path),
                                 "history_compare.log")

        cmd = [
            "npz_tool.py",
            "compare",
            target.files.bmodel_inference.path,
            reference.files.bmodel_inference.path,
            "--forall",
            "--save",
            save_file,
            "--only_key",
        ]
        exitcode, output = getstatusoutput_v2(
            " ".join(cmd),
            cwd=os.path.dirname(target.files.bmodel.path),
            shell=True,
            check=False,
        )
        if exitcode != 0:
            logger.error(f"[failed] {cmd}")
        target_recorder.add_file(history_compare_file=save_file)
        target_recorder.dump()

    @check()
    def task_cmp_metric(self, target_file: str, reference_file: str = None):
        """
        all in one to collect info for debug comparison problems.
        """
        self.task_tpu_output(target_file)

        self.task_bmodel_output(target_file)
        self.task_tpu2graph(target_file)
        self.task_final2graph(target_file)
        if reference_file and os.path.exists(reference_file):
            self.task_bmodel_output(reference_file)
            self.task_npz_bmodel_compare(target_file, reference_file)
            self.task_tpu2graph(target_file)
            self.task_final2graph(target_file)

        record = CommandRecorder(target_file, read=True)
        target = RefFile(**record._dic)
        if _all_file_no_change(target.files.tpu_addressed_svg):
            logger.info(f"see address graph in {target.files.tpu_addressed_svg.path}")
            logger.info(f"see lowered graph in {target.files.tpu_lowered_svg.path}")


class DebugPerformance(DebugBase):

    @check(depend_files=["tpu_mlir"], gen_files=["lg_log", "final_mlir"])
    def task_layer_group_log(self, target_file: str):
        """re-run tpuc-opt command with --layer-group pass and -debug-only to generate lg log"""
        recorder = CommandRecorder(target_file, read=True)
        target = RefFile(**recorder.dic)
        command = target.commands.final

        left, right = command.split(" -o ")
        lg_dir = Path(target.files.final_mlir.path).parent.joinpath(BASE_DIAGNOSITIC_NAME, "logs")
        lg_dir.mkdir(exist_ok=True)
        lg_file = lg_dir.joinpath("lg.log")
        logger.info(f"will write log to {lg_file.absolute()}")
        command = [
            left,
            "-debug-only",
            "lg_index,lg_cost,lg_results,cut_optimize",
            "-o",
            target.files.final_mlir.path.replace(".mlir", "_debug.mlir"),
        ]

        command = " ".join(command)
        time_start = time.time()
        with open(lg_file.absolute(), "w") as f:
            ret, _ = getstatusoutput_v2(
                command,
                shell=True,
                check=True,
                redirect_output=f,
                redirect_error=f,
                cwd=os.path.dirname(target.files.final_mlir.path),
            )
            time_end = time.time()
            f.write(f"\n\nTime cost: {time_end - time_start:.2f}s")

        recorder.add_file(lg_log=str(lg_file.absolute()))
        recorder.dump()

    @check(depend_tasks=[], gen_files=["simulation_dir"])
    def task_simulation(self, target_file: str):
        """simulation mode, use TPUPerf, should set env PERFAI_SIMULATION_ROOT=/path/to/TPUPerf"""
        from tools.run_bmprofile import simulation

        recorder = CommandRecorder(target_file, read=True)
        target = RefFile(**recorder.dic)
        if _all_file_no_change(target.files.bmodel):
            try:
                simulation_dir = os.path.abspath(simulation(target.files.bmodel.path))
                recorder.add_file(simulation_dir=simulation_dir)
                recorder.dump()
                logger.info("run perfweb or perfdoc to parse simulation data")
            except Exception as e:
                logger.error(f"simulation failed: {e}")

    @check(depend_tasks=[], gen_files=["bmprofile_dir"])
    def task_realprofile(self, target_file: str):
        """bmrt_test mode, export BMRUNTIME_ENABLE_PROFILE=1"""
        from tools.run_bmprofile import bmprofile

        recorder = CommandRecorder(target_file, read=True)
        target = RefFile(**recorder.dic)
        if _all_file_no_change(target.files.bmodel):
            try:
                profile_raw = bmprofile(target.files.bmodel.path)
                recorder.add_file(bmprofile_dir=os.path.abspath(profile_raw))
                recorder.dump()
                logger.info("run perfweb or perfdoc to parse bmprofile data")
            except Exception as e:
                logger.error(f"bmprofile failed: {e}")

    @check(depend_tasks=[task_realprofile, task_simulation])
    def task_profile_raw(self, target_file: str):
        """
        get raw profile data, if devices is found, use bmrt_test mode, otherwise use simulation mode(TPUPerf)
        """
        recorder = CommandRecorder(target_file, read=True)
        target = RefFile(**recorder.dic)
        if _all_file_no_change(target.files.bmprofile_dir) and _all_file_no_change(
                target.files.simulation_dir):
            raise RuntimeError("bmprofile_dir or simulation_dir both generate failed.")

    @check(depend_tasks=[task_profile_raw], gen_files=["profile_web"])
    def task_perfweb(self, target_file: str):
        from profile_helper.interface import (
            bmprofile_parse_perfAI,
            bmprofile_analyze,
        )

        recorder = CommandRecorder(target_file, read=True)
        target = RefFile(**recorder.dic)

        raw_dir = None
        if _all_file_no_change(target.files.bmprofile_dir):
            raw_dir = target.files.bmprofile_dir.path
        elif _all_file_no_change(target.files.simulation_dir):
            raw_dir = target.files.simulation_dir.path
        else:
            raise Exception("bmprofile_dir or simulation_dir not exists")

        output_dir = os.path.join(os.path.dirname(raw_dir), "profile_rich")

        perfweb_dir = os.path.join(output_dir, "PerfWeb")
        if os.path.exists(perfweb_dir):
            shutil.rmtree(perfweb_dir)
        if target.properties.chip.upper() in ["BM1688", "CV186X", "BM1690"]:
            bmprofile_parse_perfAI(raw_dir,
                                   output_dir,
                                   target.properties.chip.upper(),
                                   web=True,
                                   doc=False)
            recorder.add_file(profile_web=perfweb_dir)
        else:
            bmprofile_analyze(raw_dir, output_dir, "html")
        recorder.dump()

    @check(depend_tasks=[task_profile_raw], gen_files=["profile_csv"])
    def task_perfdoc(self, target_file: str):
        from profile_helper.interface import (
            bmprofile_parse_perfAI,
            bmprofile_analyze,
        )

        recorder = CommandRecorder(target_file, read=True)
        target = RefFile(**recorder.dic)
        if _all_file_no_change(target.files.bmprofile_dir):
            raw_dir = target.files.bmprofile_dir.path
        elif _all_file_no_change(target.files.simulation_dir):
            raw_dir = target.files.simulation_dir.path
        else:
            raise Exception("bmprofile_dir or simulation_dir not exists or is modified")

        output_dir = os.path.join(os.path.dirname(raw_dir), "profile_rich")

        os.makedirs(output_dir, exist_ok=True)
        perfdoc_dir = os.path.join(output_dir, "PerfDoc")
        if os.path.exists(perfdoc_dir):
            shutil.rmtree(perfdoc_dir)

        if target.properties.chip.upper() in ["BM1688", "CV186X", "BM1690"]:
            bmprofile_parse_perfAI(
                raw_dir,
                output_dir,
                arch=target.properties.chip.upper(),
                doc=True,
                web=False,
            )

            recorder.add_file(profile_csv=os.path.join(output_dir, "PerfDoc", "PerfAI_output.csv"))
        else:
            output_dir = os.path.join(output_dir, "PerfDoc")
            os.makedirs(output_dir, exist_ok=True)
            bmprofile_analyze(raw_dir, output_dir, "csv")
            recorder.add_file(profile_csv=os.path.join(output_dir, "detail.csv"))

        recorder.dump()

    @check(depend_tasks=[task_perfdoc, task_perfweb, "task_mlir2onnx", "task_mlirs"])
    def task_profile_rich(self, target_file: str):
        """"""

    @check(depend_files=["lg_log"], force=True)
    def task_parse_lglog2table(self, target_file: str):
        """
        use lg log to get op granularity cycle and lg cost table
        """
        from tools.logdebug_tool import op_lg_cost, cost_table

        recorder = CommandRecorder(target_file, read=True)
        target = RefFile(**recorder.dic)

        if _all_file_no_change(target.files.lg_log):
            ret = {
                "lg_profile": op_lg_cost(target.files.lg_log.path),
                "cost_table": cost_table(target.files.lg_log.path),
            }

            def _style_cost_table(x):
                color_list = [""] * len(x)
                if x.name in group_map:
                    color_list[2 + group_map[x.name]] = "background-color: #c6efce;"

                return color_list

            group_map = {}
            for group_idx, df in ret["lg_profile"].groupby("group_idx"):
                group_map[df["op_idx"].min()] = df["op_idx"].max()

            acc = 0
            group_base_offset = {}
            for group_idx, df in ret["lg_profile"].groupby("base_group_idx"):
                group_base_offset[group_idx] = acc
                acc += len(df)

            ret["cost_table"] = ret["cost_table"].style.apply(_style_cost_table, axis=1)
            dir_path = os.path.dirname(target.files.lg_log.path)
            # ret["cost_table"].to_html(os.path.join(dir_path, "cost_table.html"))
            ret["cost_table"].to_excel(os.path.join(dir_path, "cost_table.xlsx"))
            # ret['lg_profile']
            # for index, row in enumerate(ret['cost_table'].index):

            return ret

    @check(
        depend_tasks=[task_perfdoc],
        depend_files=["final_mlir", "tensor_location"],
        force=True,
    )
    def task_parse_profile2df(self, target_file: str):
        """
        use perf csv file to get op granularity cycle
        """
        recorder = CommandRecorder(target_file, read=True)
        target = RefFile(**recorder.dic)
        final_mlir, tl_file = target.files.final_mlir, target.files.tensor_location
        df = _load(tl_file.path)

        groups = {}
        parse_ret = _parse_groups(final_mlir.path)
        groups, group_with_line = parse_ret["groups"], parse_ret["group_with_line"]

        min_id = {}
        max_id = {}

        _, min_id, max_id = _parse_group_cmd_id_range(groups, df)

        perfdf = None
        if _all_file_no_change(target.files.profile_csv):
            perfdf = pd.read_csv(target.files.profile_csv.path)
            if "Core Id" not in perfdf.columns:
                perfdf["Core Id"] = 0
            if "category" in perfdf.columns:
                perfdf = perfdf[perfdf["category"].apply(
                    lambda x: "BD" in x or "GDMA" in x)].reset_index(drop=True)
                perfdf["Engine Id"] = perfdf["category"].apply(lambda x: 0 if "TPU_BD" in x else 1)
                perfdf["Cmd Id"] = (
                    perfdf["func_type"].apply(lambda x: x.split("=")[-1]).apply(float))
                perfdf["Stall Cycle"] = 0
                # breakpoint()

                perfdf.rename(
                    columns={
                        "begin_usec": "Start Cycle",
                        "end_usec": "End Cycle",
                    },
                    inplace=True,
                )

        # print(bmodel, perfcsv)
        columns = [
            "groups",
            "op_in_group",
            "max_dma_cycle_range",
            "max_tiu_cycle_range",
        ]
        all_ret = []
        # print(*columns, sep=",")
        match_tpu_op = re.compile('(tpu\.[a-zA-Z0-9]+)"')
        core_ids = sorted(set(perfdf["Core Id"]))
        left = 0
        accu = 0
        cycle_dma_acc = 0
        cycle_tiu_acc = 0
        for k, vv in groups.items():
            # if len(vv) <= 1:
            tpuops = [match_tpu_op.findall(i)[0] for i in group_with_line[k]]
            accu += len(tpuops)

            #     print("skip", vv[0], lines[vv[0] - 1])
            min_core_dic = min_id[k]
            max_core_dic = max_id[k]

            # tiu, dma, tiu1, dma1 = min_id[k]
            # tium, dmam, tium1, dmam1 = max_id[k]

            def parse_group(core_id):
                dmadf = perfdf[(perfdf["Engine Id"] == 1) & (perfdf["Core Id"] == core_id)]
                tiudf = perfdf[(perfdf["Engine Id"] == 0) & (perfdf["Core Id"] == core_id)]
                _tiu, _dma = min_core_dic[core_id]
                _tium, _dmam = max_core_dic[core_id]

                # breakpoint()
                dma_begin_cycle = 10**9
                dma_end_cycle = 0

                ret = dmadf[(dmadf["Cmd Id"] > _dma) & (dmadf["Cmd Id"] <= _dmam)]
                dma_begin_cycle = min(dma_begin_cycle, ret["Start Cycle"].min())
                dma_end_cycle = max(dma_end_cycle, ret["End Cycle"].max())
                # print(dma_end_cycle - dma_begin_cycle, dma_end_cycle, dma_begin_cycle)
                # breakpoint()
                stall_cycle = ret["Stall Cycle"].sum()
                dma_end_cycle -= stall_cycle

                tiu_begin_cycle = 10**9
                tiu_end_cycle = 0
                ret = tiudf[(tiudf["Cmd Id"] > _tiu) & (tiudf["Cmd Id"] <= _tium)]
                tiu_begin_cycle = min(tiu_begin_cycle, ret["Start Cycle"].min())
                tiu_end_cycle = max(tiu_end_cycle, ret["End Cycle"].max())
                # dma_dur = (dma_end_cycle - dma_begin_cycle) / 750000
                # tiu_dur = (tiu_end_cycle - tiu_begin_cycle) / 900000

                # min_begin = min(tiu_begin_cycle, dma_begin_cycle)
                # max_end = max(tiu_end_cycle, dma_end_cycle)
                # total_dur = max_end - min_begin

                return {
                    "dma": _dma,
                    "dma1": _dmam,
                    "tiu": _tiu,
                    "tiu1": _tium,
                    "dma_begin_cycle": dma_begin_cycle,
                    "tiu_begin_cycle": tiu_begin_cycle,
                    "dma_end_cycle": dma_end_cycle,
                    "tiu_end_cycle": tiu_end_cycle,
                    "dma_cycle_range": max(dma_end_cycle - dma_begin_cycle, 0),
                    "tiu_cycle_range": max(tiu_end_cycle - tiu_begin_cycle, 0),
                }

            def max_key(col):
                return max([ret[col] for ret in rets])

            if perfdf is not None:
                dma_repr = []
                tiu_repr = []
                rets = []
                for _core_id in core_ids:
                    ret = parse_group(_core_id)
                    if ret["dma"] != 10**9:
                        dma_repr.append(f"{ret['dma']}->{ret['dma1']}")
                    if ret["tiu"] != 10**9:
                        tiu_repr.append(f"{ret['tiu']}->{ret['tiu1']}")
                    rets.append(ret)

                dma_repr = ";".join(dma_repr)
                tiu_repr = ";".join(tiu_repr)
                # ret0 = parse_group(0)
                # ret1 = parse_group(1)

                cycle_dma_acc += max_key("dma_cycle_range")
                cycle_tiu_acc += max_key("dma_cycle_range")
                while left < accu:
                    all_ret.append({
                        "group_idx": k,
                        "op_idx": left,
                        "max_dma_cycle_range": max_key("dma_cycle_range"),
                        "cycle_dma_acc": cycle_dma_acc,
                        "max_tiu_cycle_range": max_key("tiu_cycle_range"),
                        "cycle_tiu_acc": cycle_tiu_acc,
                        "dma_repr": dma_repr,
                        "tiu_repr": tiu_repr,
                    })

                    left += 1

        return {"perf_profile": pd.DataFrame.from_records(all_ret)}

    @check()
    def task_cmp_perf(self, target_file: str, reference_file: str = None):
        """
        combine lg cost table and perf csv file and their comparison results
        """

        def _find_acc_col_name(df: pd.DataFrame):
            if "lg_cost_accumulate" in df.columns:
                return "lg_cost_accumulate"
            if "cycle_dma_acc" in df.columns:
                return "cycle_dma_acc"
            raise NotImplementedError()

        def _find_op_cycle_col_name(df: pd.DataFrame):
            if "lg_group_cost" in df.columns:
                return "lg_group_cost"
            if "max_dma_cycle_range" in df.columns:
                return "max_dma_cycle_range"
            raise NotImplementedError()

        def _merge_lgdf(left: pd.DataFrame, right: pd.DataFrame, same_commit=False):
            """
            same commit means lg cost vs. perf cost
            not same commit means lg cost vs. lg cost and perf cost vs. perf cost
            """
            if same_commit:
                right = right.drop(columns=["group_idx", "op_idx"])
            left[" "] = ""

            ret = pd.concat([left, right], ignore_index=True, axis=1)
            if not same_commit:
                ret.columns = [
                    *left.columns.tolist(),
                    *[f"ref_{i}" for i in right.columns.tolist()],
                ]
            else:
                ret.columns = [*left.columns.tolist(), *right.columns.tolist()]

            scale = 1
            if same_commit:
                # check lg profile vs. real/perfai profile
                left_acc_col = _find_op_cycle_col_name(left)
                right_acc_col = _find_op_cycle_col_name(right)
                ret["ratio"] = (left[left_acc_col] -
                                right[right_acc_col] / 100) / (right[right_acc_col] / 100)
                return ret

            else:
                # check diff commit
                left_acc_col = _find_acc_col_name(left)
                right_acc_col = _find_acc_col_name(right)
                ret["residual"] = left[left_acc_col] - right[right_acc_col]

                mid_column_index = ret.columns.tolist().index(" ")

                def color_diff(x):
                    color_pool = [
                        "background-color: #f4c1c7;",
                        "background-color: #c6efce;",
                    ]
                    font_pool = ["font-weight: bold;", ""]
                    color_list = [""] * len(x.index)
                    x_index = ret.columns.tolist().index(x.name)
                    if x_index < mid_column_index:
                        name = "group_idx"
                    elif x_index > mid_column_index:
                        name = "ref_group_idx"
                    else:
                        return color_list
                    color_list[0] = color_pool[0]
                    for i, xindx in enumerate(x.index[1:], start=1):
                        cur = ret.iloc[xindx][name]
                        pre = ret.iloc[xindx - 1][name]
                        if cur != pre:
                            color_pool[0], color_pool[1] = color_pool[1], color_pool[0]

                        color_list[i] = color_pool[0]

                        # font_pool[0], font_pool[1] = font_pool[1], font_pool[0]

                    return color_list

                ret_style = ret.style.apply(color_diff, axis=0)
                return ret_style

        target_ret = {}
        target = RefFile(**CommandRecorder(target_file, read=True)._dic)
        # target.files.

        target_ret.update(self.task_parse_lglog2table(target_file))
        target_ret.update(self.task_parse_profile2df(target_file))
        target_ret["target lg vs. perf"] = _merge_lgdf(target_ret["lg_profile"],
                                                       target_ret["perf_profile"],
                                                       same_commit=True)

        reference_ret = {}
        if reference_file and os.path.exists(reference_file):
            reference_ret.update(self.task_parse_lglog2table(reference_file))
            reference_ret.update(self.task_parse_profile2df(reference_file))
            reference_ret["ref lg vs. perf"] = _merge_lgdf(
                reference_ret["lg_profile"],
                reference_ret["perf_profile"],
                same_commit=True,
            )

        if "lg_profile" in reference_ret:
            target_ret["target lg vs. ref lg"] = _merge_lgdf(target_ret["lg_profile"],
                                                             reference_ret["lg_profile"])

        if "perf_profile" in reference_ret:
            target_ret["target perf vs. ref perf"] = _merge_lgdf(target_ret["perf_profile"],
                                                                 reference_ret["perf_profile"])

        target_xlsx = os.path.join(os.path.dirname(target_file), "debugit_perf.xlsx")
        with pd.ExcelWriter(target_xlsx) as writer:

            for k, df in target_ret.items():
                df.to_excel(writer, sheet_name=k)
            for k, df in reference_ret.items():
                df.to_excel(writer, sheet_name=f"ref_{k}")
        logger.info(f"{target_xlsx} generated")

    @check(depend_tasks=[task_layer_group_log, task_profile_raw, task_profile_rich])
    def task_perf_data(self, target_file: str):
        """
        generate lg log, perf csv and web if not exists
        """


# def context(func: Callable):
#     def process_ret(ret):
#         if isinstance(ret, pd.DataFrame):
#             return ret.to_csv(sys.stdout, index=False, sep=",")
#         elif isinstance(ret, dict):
#             nret = {}
#             for k, v in ret.items():
#                 nret[k] = process_ret(v)
#             return nret
#         elif isinstance(ret, list):
#             return [process_ret(v) for v in ret]
#         else:
#             return ret

#     def wrapper(*args, **kwargs):
#         logger.info(f"start {func.__name__}")

#         ret = func(*args, **kwargs)
#         logger.info(f"end {func.__name__}")

#         return process_ret(ret)

#     return wrapper
