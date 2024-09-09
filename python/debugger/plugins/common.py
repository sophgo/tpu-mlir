# ==============================================================================
#
# Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
#
# TPU-MLIR is licensed under the 2-Clause BSD License except for the
# third-party components.
#
# ==============================================================================
import re
from collections import OrderedDict
import collections
from dataclasses import dataclass
from typing import List, Dict, Optional
import pickle
import os
from rich.progress import (
    Progress as Progressbar,
    TimeRemainingColumn,
    TimeElapsedColumn,
    TaskProgressColumn,
    TextColumn,
    ProgressColumn,
    BarColumn,
)

import pandas as pd
import numpy as np

from ..final_mlir import CMD, FinalMlirIndex, TLValue

from ..target_common.op_support import BaseTpuCmd
from ..target_common import CMDType, ValueRef
from ..tdb_support import (
    TdbCmdBackend,
    TdbPlugin,
    Displays,
    TdbStatus,
    TdbPluginCmd,
    Watchpoint,
    complete_file,
)
from ..target_1688.context import BM1688Context
from ..target_1690.context import BM1690Context
from ..target_2380.context import SG2380Context

_invalid_fc = r"[+?@#$&%*()=;|,<>: +" r"\^\/\t\b\[\]\"]+"


def to_filename(basename):
    """Replace invalid characters in a basename with an underscore."""
    return re.sub(_invalid_fc, "_", basename)


def max_with_none(*args):
    args = [i for i in args if i is not None]
    if len(args) == 1:
        return args[0]
    if len(args) == 0:
        return 0

    return max(args)


@dataclass
class ValueView:
    value: TLValue
    index: int
    loc_index: int
    loc_name: str
    cmd_point: int
    file_line: int

    @property
    def is_operand(self):
        return isinstance(self, Operand)


class Result(ValueView):
    pass


class Operand(ValueView):
    pass


class ReloadPlugin(TdbPlugin, TdbPluginCmd):
    name = "reload"

    def do_mlir(self, arg):
        """
        reload mlir <final.mlir> <tensor_location.json>
        """
        res = arg.split(" ")
        if len(res) != 2:
            self.tdb.error("reload mlir <final.mlir> <tensor_location.json>")
            return

        final_mlir, tensor_location = res
        self.tdb.final_mlir_fn = final_mlir
        self.tdb.tensor_loc_file = tensor_location
        self.tdb.do_start()

    def do_input(self, input):
        """
        reload input input_fn
        """
        self.tdb.input_data_fn = input
        self.tdb.do_start()

    complete_mlir = complete_file
    complete_input = complete_file


class DumpIndex(TdbPlugin, TdbPluginCmd):
    name = "dump"

    def do_index(self, arg):
        """
        reload mlir <final.mlir> <tensor_location.json>
        """
        self.tdb.index_df.to_excel("index.xlsx")


class FinalMlirIndexPlugin(TdbPlugin):
    """
    append final-mlir indexs by extending tdb.index_df columns.

    executed_id, loc_indexs

    """

    name = "final-mlir"
    data_frame = None

    def __init__(self, tdb: TdbCmdBackend) -> None:
        super().__init__(tdb)

    def __str__(self) -> str:
        flag = "√" if self.enabled else "x"
        return f"{self.name}({flag})"

    @property
    def enabled(self):
        tdb = self.tdb
        return tdb.final_mlir_fn is not None and tdb.tensor_loc_file is not None

    @property
    def index_df(self):
        return self.tdb.index_df

    @property
    def final_mlir_fn(self):
        return self.tdb.final_mlir_fn

    @property
    def tensor_loc_file(self):
        return self.tdb.tensor_loc_file

    def after_load(self, tdb: TdbCmdBackend):
        if self.enabled:
            self._build_index(tdb)

    def _build_index(self, tdb: TdbCmdBackend):
        if tdb.cache_mode in {'online', 'generate'}:
            ret = self._build_mlir_loc(tdb)
            if tdb.cache_mode == 'generate':
                with open(f"{self.tdb.final_mlir_fn}.tdb_cache.pickle", 'wb') as w:
                    pickle.dump(ret, w)
        elif tdb.cache_mode == 'offline':
            with open(f"{self.tdb.final_mlir_fn}.tdb_cache.pickle", 'rb') as r:
                ret = pickle.load(r)
        else:
            raise NotImplementedError(tdb.cache_mode)

        tdb.global_layer_line = ret["global_layer_line"]
        loc_df = ret['df']
        self.final_mlir: FinalMlirIndex = ret['final_mlir']

        tdb.index_df = pd.merge(tdb.op_df, loc_df, how="outer")
        tdb.index_df = tdb.index_df.set_index("cmd_index", drop=True)
        # replace all NaN values with zeros
        tdb.index_df["loc_index"] = tdb.index_df["loc_index"].fillna(-1)
        # convert 'loc_index' column from float to integer
        tdb.index_df["loc_index"] = tdb.index_df["loc_index"].astype(int)
        FinalMlirIndexPlugin.data_frame = tdb.index_df

    def _build_mlir_loc(self, tdb: TdbCmdBackend):
        final_mlir = FinalMlirIndex.from_context(self.final_mlir_fn, self.tensor_loc_file)
        last_af_point = 0
        index_dic = tdb.op_df.to_dict("index")

        def find_point(key):
            ret = index_dic[key]
            return ret['executed_id']

        def assemble_tuple_key(subnet_id, core_id, cmd_id, cmd_type):
            if cmd_type == CMDType.tiu:
                return (subnet_id, cmd_id, None, core_id)
            elif cmd_type == CMDType.dma:
                return (subnet_id, None, cmd_id, core_id)
            else:
                raise RuntimeError("Not Supported CMDType!")

        # when cmd_point reach int `point`(index start from 1)
        # it means the atomic_cmd in cmditer[point-1] has been executed
        # data-checker need to compare loc operands before execute bf_point
        # and after execute af_point

        loc_records = []
        global_layer_line = collections.Counter()
        for loc_index, loc in enumerate(final_mlir.loc.tensor_loc):
            if loc.slice_all:
                global_layer_line[loc.file_line] += 1

            if loc.tiu_dma_id_before == loc.tiu_dma_id_after:
                # no cmd operation, like reshape
                continue

            # the tiu/dma cmd-id state before execution this loc
            # we need to find the pointer

            (
                subnet_id,
                tiu_before,
                dma_before,
                core_id,
            ) = loc.tuple_key_before
            (
                subnet_id,
                tiu_after,
                dma_after,
                core_id,
            ) = loc.tuple_key_after
            if tiu_after > tdb.tiu_max[core_id] or dma_after > tdb.dma_max[core_id]:
                continue

            tiu_point = dma_point = None
            if tiu_before > 0:
                tiu_point = find_point((subnet_id, tiu_before, None, core_id))
            if dma_before > 0:
                dma_point = find_point((subnet_id, None, dma_before, core_id))
            bf_point = max_with_none(tiu_point, dma_point, last_af_point) + 1
            operands_tmp = collections.defaultdict(list)
            operands_tmp[tdb.cmditer[bf_point - 1].tuple_key].extend(
                Operand(opd, opd_index, loc_index, opd.name, bf_point, loc.file_line)
                for opd_index, opd in enumerate(loc.operands) if opd)

            tiu_point = dma_point = None
            if tiu_after > 0:
                tiu_point = find_point((subnet_id, tiu_after, None, core_id))
            if dma_after > 0:
                dma_point = find_point((subnet_id, None, dma_after, core_id))
            last_af_point = af_point = max_with_none(tiu_point, dma_point)
            results_tmp = collections.defaultdict(list)
            results_tmp[tdb.cmditer[af_point - 1].tuple_key].extend(
                Result(opd, opd_index, loc_index, opd.name, af_point, loc.file_line)
                for opd_index, opd in enumerate(loc.results) if opd)

            for i in range(tiu_before + 1, tiu_after + 1):
                record = {
                    "loc_index": loc.loc_index,
                    "line-num": loc.file_line,
                    "subnet_id": subnet_id,
                    "core_id": core_id,
                    "cmd_id": i,
                    "cmd_type": CMDType.tiu,
                    "operands": operands_tmp[assemble_tuple_key(subnet_id, core_id, i,
                                                                CMDType.tiu)],
                    "results": results_tmp[assemble_tuple_key(subnet_id, core_id, i, CMDType.tiu)],
                }
                loc_records.append(record)

            for j in range(dma_before + 1, dma_after + 1):
                record = {
                    "loc_index": loc.loc_index,
                    "line-num": loc.file_line,
                    "subnet_id": subnet_id,
                    "core_id": core_id,
                    "cmd_id": j,
                    "cmd_type": CMDType.dma,
                    "operands": operands_tmp[assemble_tuple_key(subnet_id, core_id, j,
                                                                CMDType.dma)],
                    "results": results_tmp[assemble_tuple_key(subnet_id, core_id, j, CMDType.dma)],
                }
                loc_records.append(record)

        loc_df = pd.DataFrame.from_records(loc_records)

        return {"df": loc_df, "global_layer_line": global_layer_line, "final_mlir": final_mlir}

    def get_mlir_by_point(self, point=None) -> Optional[str]:
        """NOTE: file-line in tensor_location.json starts from 1"""
        file_line = self.tdb.index_df.loc[
            self.tdb.index_df["executed_id"] == point + 1, "line-num"
        ].item()

        file_line = int(file_line)

        return self.final_mlir.lines[file_line - 1]

    def get_mlir_context_by_point(
        self, point=None, pre=2, next=2
    ) -> Optional[List[str]]:
        file_line = self.tdb.index_df.loc[
            self.tdb.index_df["executed_id"] == point + 1, "line-num"
        ].item()
        return self.final_mlir.lines[max(0, file_line - 1 - pre) : file_line - 1 + next]

    def get_locindex_by_atomic(self, point=None) -> Optional[int]:
        """
        N cmds have N+1 positions,
        use tdb.cmd_point other than cmd2index to get current point
        """
        if point is None:
            point = self.tdb.cmd_point

        loc_index = self.tdb.index_df.loc[
            self.tdb.index_df["executed_id"] == point + 1, "loc_index"
        ].item()

        if np.isnan(loc_index):
            return None
        return int(loc_index)

    def get_loc_by_point(self, point=None) -> Optional[CMD]:
        loc_index = self.get_locindex_by_atomic(point)
        if loc_index is None:
            return None
        return self.final_mlir.loc[loc_index]

    def get_loc_context_by_point(
        self, point=None, pre=2, next=2
    ) -> Optional[List[CMD]]:
        loc_index = self.get_locindex_by_atomic(point)
        if loc_index is None:
            return None
        return self.final_mlir.loc[max(0, loc_index - pre) : loc_index + next]


class DisplayPlugin(TdbPlugin, TdbPluginCmd):
    name = "display"

    def __init__(self, tdb: TdbCmdBackend) -> None:
        super().__init__(tdb)
        self.displays = Displays.get_instance()
        tdb.complete_display = tdb._complete_expression

    def do_delete(self, arg):
        """remove display"""
        pass

    def default(self, arg):
        """
        display arg info after each break

         - address:  4295028736@40
        """
        try:
            eval(arg)
        except Exception as e:
            self.error(f"Can not add display {arg}")
            self.error(e)
            return
        item_id = self.displays.add_display(arg)
        self.message(f"{item_id} {eval(arg)}")
        # self.plugins.after_stop(self)

    def after_stop(that, tdb: TdbCmdBackend):
        self = tdb
        table = []
        for k, dis in self.displays.display.items():
            try:
                table.append([f"%{k}:", str(eval(dis.expr))])
            except Exception:
                continue
        if len(table) > 0:
            df = pd.DataFrame(table)
            tdb.message(df.to_string(index=False, header=False))


class PrintPlugin(TdbPlugin, TdbPluginCmd):
    name = "print"
    func_names = ["p", "print"]

    def __init__(self, tdb: TdbCmdBackend) -> None:
        super().__init__(tdb)
        self.dump_all = False

    def do_in(self, arg):
        try:
            cmd = self.tdb.get_cmd()
        except StopIteration:
            self.tdb.message("no cmd next.")
            return
        if arg == "":
            self.tdb.message(cmd.operands)
            return

        try:
            index = int(arg)
            if cmd.cmd_type == CMDType.cpu:
                if cmd.cmd_id == 0:
                    data = self.tdb.memory.get_data(cmd.operands[index].to_ref())
                else:
                    data = self.tdb.memory.get_cpu_data(cmd.cmd_id)[cmd.operands[index]]
            elif cmd.cmd_type.is_static():
                if cmd.operands[index].is_scalar:
                    data = cmd.operands[index].data
                else:
                    data = self.tdb.memory.get_data(cmd.operands[index].to_ref(core_id=cmd.core_id))
            else:
                self.tdb.error("")
                return
            print(data)
        except (IndexError, SyntaxError, ValueError) as e:
            self.tdb.error(e)

    def do_next(self, arg):
        try:
            cmd = self.tdb.get_cmd()
            self.tdb.message(cmd)
        except StopIteration:
            self.tdb.error("no cmd next.")

    do_op = do_next

    def do_pre(self, arg):
        try:
            op = self.tdb.get_precmd()
            self.tdb.message(op)
        except StopIteration:
            self.tdb.message("no cmd pre.")
            return

    def do_out(self, arg):
        try:
            cmd = self.tdb.get_precmd()
        except StopIteration:
            self.tdb.message("no cmd pre.")
            return

        if arg == "":
            self.tdb.message(cmd.results)
            return

        try:
            index = int(arg)
            if cmd.cmd_type == CMDType.cpu:
                data = self.tdb.memory.get_cpu_data(cmd.cmd_id)[index]
            elif cmd.cmd_type.is_static():
                if cmd.results[index].is_scalar:
                    data = cmd.results[index].data
                else:
                    data = self.tdb.memory.get_data(cmd.results[index].to_ref(core_id=cmd.core_id))
            else:
                self.tdb.error("")
                return
            print(data)
        except (IndexError, SyntaxError, ValueError) as e:
            self.tdb.error(e)

    def do_w(self, arg):
        return self.do_watch(arg)

    def do_watch(self, arg):
        if arg == "":
            self.tdb.message("Please enter the watch index to print")
            return

        watches = self.tdb.watches
        watchid2value = self.tdb.watchid2value

        arg_values = arg.split()
        index, watchpoint_type = (arg_values + [None])[:2]
        index = int(index)
        try:
            if watchpoint_type == "now" or not watchpoint_type:
                watchpoint: Watchpoint = watches[watchid2value[index][0]]
            elif watchpoint_type == "old":
                print(watchid2value[index][1])
                return
        except Exception:
            self.tdb.message(
                "Please input correct watch index, enter `w` or `watch` to checkout current watchpoints"
            )
            return

        try:
            if watchpoint.cmd_type == CMDType.cpu:
                if watchpoint.cmd_id == 0:
                    data = self.tdb.memory.get_data(ValueRef(watchpoint.value))
                else:
                    data = self.tdb.memory.get_cpu_data(watchpoint.cmd_id)[watchpoint.value]
            elif watchpoint.cmd_type.is_static():
                if watchpoint.value.is_scalar:
                    data = watchpoint.value.data
                else:
                    data = self.tdb.memory.get_data(ValueRef(watchpoint.value, core_id=watchpoint.core_id))
            else:
                self.tdb.error("")
                return
            print(data)
        except (IndexError, SyntaxError, ValueError) as e:
            self.tdb.error(e)

    def after_step(self, tdb: TdbCmdBackend):
        if self.dump_all:
            self.dump_current()

        # from debugger.target_1688.regdef import DMA_tensor_0x000__reg
        # if tdb.cmd_point > 1 and isinstance(tdb.get_precmd().reg, DMA_tensor_0x000__reg):
        #     cmd = tdb.get_precmd()
        #     ipt = tdb.memory.get_data(cmd.operands[0].to_ref(core_id=cmd.core_id))
        #     opt = tdb.memory.get_data(cmd.results[0].to_ref(core_id=cmd.core_id))
        #     if not (ipt == opt).all():
        #         breakpoint()
        #     else:
        #         tdb.message("succeed dma ops")

    def dump_current(self):
        if self.tdb.context.using_cmodel:
            filename = f"info_dump_{self.tdb.cmd_point}_cmodel.npz"
        else:
            filename = f"info_dump_{self.tdb.cmd_point}_device.npz"

        datas = {}
        try:
            cmd = self.tdb.get_cmd()
            if cmd.cmd_type == CMDType.cpu:
                data = self.tdb.memory.get_cpu_data(cmd.cmd_id)
                for i, d in enumerate(data):
                    datas[i] = d
            elif cmd.cmd_type.is_static():
                for i, result in enumerate(cmd.operands):
                    data = self.tdb.memory.get_data(result.to_ref(core_id=cmd.core_id))
                    datas[f"i_{i}"] = data
            else:
                self.tdb.error("")
                return
        except StopIteration:
            self.tdb.message("no cmd pre.")

        try:
            cmd = self.tdb.get_precmd()
            if cmd.cmd_type == CMDType.cpu:
                data = self.tdb.memory.get_cpu_data(cmd.cmd_id)
                for i, d in enumerate(data):
                    datas[i] = d
            elif cmd.cmd_type.is_static():
                for i, result in enumerate(cmd.results):
                    data = self.tdb.memory.get_data(result.to_ref(core_id=cmd.core_id))
                    datas[f"o_{i}"] = data
            else:
                self.tdb.error("")
                return
        except StopIteration:
            self.tdb.message("no cmd pre.")

        np.savez(filename, datas)
        self.tdb.message(f"dump in {filename}")

    def do_dump_current(self, args):
        self.dump_current()

    def after_start(self, tdb: TdbCmdBackend):
        try:
            tdb.message(tdb.get_cmd())
        except StopIteration:
            pass

    def after_stop(self, tdb: TdbCmdBackend):
        try:
            tdb.message(tdb.get_cmd())
        except StopIteration:
            pass

    def do_dump_all(self, args):
        args_ = args.upper().strip()
        if args_ in {"TRUE", "FALSE"}:
            self.dump_all = args_ == 'TRUE'
            self.tdb.message(f"dump_all = {self.dump_all}")
        else:
            self.tdb.error(f"should use TRUE(true, True) or FALSE(False, false), got {args}")

    def do_replace_lmem(self, args):
        if not os.path.exists(args):
            self.tdb.message(f"not found {args} in working directory :{os.getcwd()}")
            return

        cmodel_lmem = np.load(args, allow_pickle=True)["mem"].view(np.float32)

        self.tdb.memory._set_local_mem(cmodel_lmem, 0)
        self.tdb.message("replace succeed")

    def do_dump_lmem(self, args="0"):
        self.tdb.runner.memory._load_local_mem(0)
        mem = self.tdb.runner.memory.LMEM[0]
        print(mem)
        if self.tdb.context.using_cmodel:
            filename = f"dump_lmem_{self.tdb.cmd_point}_cmodel.npz"
        else:
            filename = f"dump_lmem_{self.tdb.cmd_point}_device.npz"

        np.savez(filename, **{'mem': mem})
        self.tdb.message(f"dump {filename}.")


class ProgressPlugin(TdbPlugin):
    name = "progress"

    def after_load(self, tdb: TdbCmdBackend):
        columns: List["ProgressColumn"] = []
        columns.extend(
            (
                TextColumn("{task.description}"),
                BarColumn(
                    style="bar.back",
                    complete_style="bar.complete",
                    finished_style="bar.finished",
                    pulse_style="bar.pulse",
                ),
                TaskProgressColumn(show_speed=True),
                TimeRemainingColumn(elapsed_when_finished=True),
                TimeElapsedColumn(),
            )
        )

        progress = Progressbar(
            *columns,
            auto_refresh=True,
            console=None,
            transient=True,
            get_time=None,
            refresh_per_second=10,
            disable=False,
        )
        self.progress_id = progress.add_task("progress", total=len(tdb.cmditer))
        self.progress = progress

        self.visited_subnet = set()
        self.progress.stop()

    def after_step(self, tdb: TdbCmdBackend):
        if tdb.status != TdbStatus.RUNNING:
            return

        self.progress.start()
        (subnet_id, tiu_id, dma_id, core_id) = tdb.get_cmd().tuple_key
        if subnet_id not in self.visited_subnet:
            self.progress.print(f"run subnet {subnet_id}")
            self.visited_subnet.add(subnet_id)

        if tiu_id is None:
            tiu_id = "-"
        if dma_id is None:
            dma_id = "-"

        self.progress.update(
            self.progress_id,
            description=f"{tdb.cmd_point} {tiu_id}/{dma_id}",
            completed=tdb.cmd_point,
        )
        self.progress.refresh()

    def after_stop(self, tdb: TdbCmdBackend):
        self.progress.stop()


class AutoStaticCheck(TdbPlugin, TdbPluginCmd):
    name = "static-check"

    def default(self, args: str):
        if args.strip() in {"?", ""}:
            self.tdb.message(self.tdb.checker)
            return
        elif self.tdb.status == TdbStatus.UNINIT:
            self.tdb.error("do check after load context, type s/start to load")
            return
        for arg in args.split(","):
            self.tdb.checker.do_checker(arg)
            self.tdb.message(f"[DONE] {arg}")

    def after_load(self, tdb: TdbCmdBackend):
        for check_name in tdb.extra_check:
            tdb.checker.do_checker(check_name)
            tdb.message(f"[DONE] {check_name}")
