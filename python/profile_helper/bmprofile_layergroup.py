"""
tpuc-opt -debug-only=layer-group  &> profile.log
then tpu_profile.py --mode layer-group log_file
"""
import math
import json
import pandas as pd
import re
from pathlib import Path
import os
import shutil


def parse_final_tl(bmodel):
    dirname = os.path.dirname(bmodel)
    basename = os.path.basename(bmodel)
    if basename == "final.mlir":
        basename == "compilation.bmodel"

    if basename == "compilation.bmodel":
        tl_file = os.path.join(dirname, "tensor_location.json")
        mlir_file = os.path.join(dirname, "final.mlir")
    else:
        context_dir = os.path.join(dirname, os.path.splitext(basename)[0])
        tl_file = os.path.join(context_dir, "tensor_location.json")
        mlir_file = os.path.join(context_dir, "final.mlir")

        if not os.path.exists(tl_file):
            os.makedirs(context_dir, exist_ok=True)
            shutil.copy(f"{bmodel}.json", tl_file)
            shutil.copy(f"{bmodel}", f"{context_dir}/compilation.bmodel")

    return mlir_file, tl_file


def parse_profile_log(profile_log):
    total_content = Path(profile_log).read_text()

    # base_group_idx -> start_idx -> end_idx -> cost
    cost_table = dict()

    kv_pattern = re.compile(r";\s*(\w+)\s*=\s*(['\"]?)([\w\s/\".]+)\2")

    for line in total_content.splitlines():
        ret = kv_pattern.findall(line)
        if not ret:
            continue
        dic = {}
        for k, _, v in ret:
            try:
                dic[k] = int(v)
            except Exception:
                dic[k] = v

        if "final_group_idx" in dic:
            cost_table[(dic["final_group_idx"], dic["start_idx"], dic["end_idx"])] = dic["group_cost"]

    match_cut_results = re.compile(r"base group\[([0-9]+)\] cut results: (.*)")
    cut_results = match_cut_results.findall(total_content)

    requirement = {}
    for base_group_idx, cut_result in cut_results:
        requirement[int(base_group_idx)] = []
        start_idx = 0
        for end_idx in cut_result.split(","):
            end_idx = end_idx.strip()
            if len(end_idx) == 0:
                continue
            end_idx = int(end_idx)
            requirement[int(base_group_idx)].append((start_idx, end_idx))
            start_idx = end_idx + 1

    record = []

    index = 0
    for base_group, groups in requirement.items():
        for start_idx, end_idx in groups:
            try:
                cost = cost_table[base_group, start_idx, end_idx]
            except:
                cost = -1
            record.append({
                'group_index': index,
                'base_group': base_group,
                "start_idx": start_idx,
                "end_idx": end_idx,
                "est_cycle": int(cost),
                "est_time(us)": cost / 900,
            })
            index += 1
    return record


def _load_tensor_location(tensor_location: str):
    location = json.loads(Path(tensor_location).read_text())
    df = pd.DataFrame.from_records(location)
    return df


def parse_perfdoc(perf_doc_csv, bmodel):
    final_mlir, tl_file = parse_final_tl(bmodel)
    df = _load_tensor_location(tl_file)
    group = None
    core_group = None
    lines = Path(final_mlir).read_text().split("\n")

    groups = {}
    group_line = []
    for line_no, line in enumerate(lines, start=1):

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
        elif len(group_line) > 0:
            groups[len(groups)] = group_line.copy()
            group_line.clear()
        elif all([i not in line for i in ["tpu.Weight", "top.None", "top.Input"]]):
            if "tpu." in line and line.startswith("      %"):
                # print(line)
                groups[len(groups)] = [line_no]
                # print(line)
                print(len(groups))

    lino_group_id = {}
    min_id = {}
    max_id = {}

    def drop_nan(val, default):
        if math.isnan(val):
            val = default
        return val

    for k, v in groups.items():

        for vv in v:
            lino_group_id[vv] = k
            fdf = df[(df["core_id"] == 0) & (df["file-line"] == vv)]
            fdf1 = df[(df["core_id"] == 1) & (df["file-line"] == vv)]
            # fdf = df[(df["file-line"] == vv)]

            # try:
            #     assert len(fdf) >= 1, f"{vv}, {lines[vv - 1]}"
            # except:
            #     continue
            min_tiu, min_dma, min_tiu1, min_dma1 = min_id.setdefault(
                k,
                (10**9, 10**9, 10**9, 10**9),
            )
            tiu = fdf["tiu_dma_id(before)"].apply(lambda x: x[0]).min()
            dma = fdf["tiu_dma_id(before)"].apply(lambda x: x[1]).min()
            tiu1 = fdf1["tiu_dma_id(before)"].apply(lambda x: x[0]).min()
            dma1 = fdf1["tiu_dma_id(before)"].apply(lambda x: x[1]).min()

            min_id[k] = (
                min(drop_nan(tiu, 10**9), min_tiu),
                min(drop_nan(dma, 10**9), min_dma),
                min(drop_nan(tiu1, 10**9), min_tiu1),
                min(drop_nan(dma1, 10**9), min_dma1),
            )

            max_tiu, max_dma, max_tiu1, max_dma1 = max_id.setdefault(k, (0, 0, 0, 0))
            tiu = fdf["tiu_dma_id(after)"].apply(lambda x: x[0]).max()
            dma = fdf["tiu_dma_id(after)"].apply(lambda x: x[1]).max()
            tiu1 = fdf1["tiu_dma_id(after)"].apply(lambda x: x[0]).max()
            dma1 = fdf1["tiu_dma_id(after)"].apply(lambda x: x[1]).max()
            max_id[k] = (
                max(drop_nan(tiu, 0), max_tiu),
                max(drop_nan(dma, 0), max_dma),
                max(drop_nan(tiu1, 0), max_tiu1),
                max(drop_nan(dma1, 0), max_dma1),
            )

    perfdf = None
    if perf_doc_csv is not None:
        perfdf = pd.read_csv(perf_doc_csv)

    print(bmodel, perf_doc_csv)

    record = []
    for k, vv in groups.items():
        # if len(vv) <= 1:
        #     print("skip", vv[0], lines[vv[0] - 1])
        tiu, dma, tiu1, dma1 = min_id[k]
        tium, dmam, tium1, dmam1 = max_id[k]

        def parse_group(core_id):
            dmadf = perfdf[(perfdf["Function Type"].apply(lambda x: "DMA" in x))
                           & (perfdf["Core Id"] == core_id)]
            tiudf = perfdf[(perfdf["Function Type"].apply(lambda x: "DMA" not in x))
                           & (perfdf["Core Id"] == core_id)]

            if core_id == 1:
                _tiu, _dma, _tium, _dmam = tiu1, dma1, tium1, dmam1
            else:
                _tiu, _dma, _tium, _dmam = tiu, dma, tium, dmam

            dma_begin_cycle = 10**9
            dma_end_cycle = 0

            ret = dmadf[(dmadf["Cmd Id"] > _dma) & (dmadf["Cmd Id"] <= _dmam)]
            dma_begin_cycle = min(dma_begin_cycle, ret["Start Cycle"].min())
            dma_end_cycle = max(dma_end_cycle, ret["End Cycle"].max())
            stall_cycle = ret['Stall Cycle'].sum()
            dma_end_cycle -= stall_cycle

            tiu_begin_cycle = 10**9
            tiu_end_cycle = 0
            ret = tiudf[(tiudf["Cmd Id"] > _tiu) & (tiudf["Cmd Id"] <= _tium)]
            tiu_begin_cycle = min(tiu_begin_cycle, ret["Start Cycle"].min())
            tiu_end_cycle = max(tiu_end_cycle, ret["End Cycle"].max())

            return {
                "dma": _dma,
                "dma1": _dmam,
                "tiu": _tiu,
                "tiu1": _tium,
                "dma_begin_cycle": dma_begin_cycle,
                "tiu_begin_cycle": tiu_begin_cycle,
                "dma_end_cycle": dma_end_cycle,
                "tiu_end_cycle": tiu_end_cycle,
                "stall_cycle": stall_cycle,
                "dma_cycle_range": max(dma_end_cycle - dma_begin_cycle, 0),
                "tiu_cycle_range": max(tiu_end_cycle - tiu_begin_cycle, 0),
            }

        if perfdf is not None:
            ret0 = parse_group(0)
            ret1 = parse_group(1)

            record.append({
                "group_index":
                k,
                "c0_dma_id_begin":
                ret0['dma'],
                "c1_cmd_id_begin":
                ret1['dma'],
                "c0_tiu_id_begin":
                ret0['tiu'],
                "c1_tiu_id_begin":
                ret1['tiu'],
                "c0_dma_id_end":
                ret0['dma1'],
                "c1_cmd_id_end":
                ret1['dma1'],
                "c0_tiu_id_end":
                ret0['tiu1'],
                "c1_tiu_id_end":
                ret1['tiu1'],
                "c0-Asic-Cycle":
                max(ret0['dma_end_cycle'] - ret0['dma_begin_cycle'], 0),
                "c1-Asic-Cycle":
                max(ret1['dma_end_cycle'] - ret1['dma_begin_cycle'], 0),
                "c0-Asic-Cycle-No-Stall":
                max(ret0['dma_end_cycle'] - ret0['dma_begin_cycle'] - ret0['stall_cycle'], 0),
                "c1-Asic-Cycle-No-Stall":
                max(ret1['dma_end_cycle'] - ret1['dma_begin_cycle'] - ret1['stall_cycle'], 0),
                "c0-Time(us)":
                max((ret0['dma_end_cycle'] - ret0['dma_begin_cycle']) / 750, 0),
                "c1-Time(us)":
                max((ret1['dma_end_cycle'] - ret1['dma_begin_cycle']) / 750, 0),
            })
        # else:
        #     print(
        #         f"group({k}),",
        #         f"{tiu}->{tiu1}",
        #         f"{dma}->{dma1}",
        #     )
    return record


class BMProfileLLayerGroup:

    def __init__(self, log_file: str, perfdoc, bmodel) -> None:
        self.log_file = log_file
        self.perfdoc = perfdoc
        self.bmodel = bmodel

    def parse(self):
        a = parse_perfdoc(perf_doc_csv=self.perfdoc, bmodel=self.bmodel)
        b = parse_profile_log(self.log_file)
        real = pd.DataFrame.from_records(a)
        est = pd.DataFrame.from_records(b)
        breakpoint()
        
        full = est.merge(real, on='group_index', how='outer')

        full['Max-Time(us)'] = full[['c0-Time(us)', 'c1-Time(us)']].apply(max, axis=1)
        full['Bias'] = (full['est_time(us)'] - full['Max-Time(us)']) / full['Max-Time(us)']
        full['Abs-Bias'] = full['Bias'].apply(abs)
        print((full.to_string()))
        Path("lg.text").write_text(full.to_string())
