#!/usr/bin/env python3
# ==============================================================================
#
# Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
#
# TPU-MLIR is licensed under the 2-Clause BSD License except for the
# third-party components.
#
# ==============================================================================
import fire
from pathlib import Path
import re
import pandas as pd
from itertools import combinations

kv_pattern = re.compile(r";\s*([\w/.]+)\s*=\s*(['\"]?)([:\w\s/\-\".]+)\2")


def comsume_in_main(func):
    def wrapper(*args, **kwargs):
        iterable = func(*args, **kwargs)
        if isinstance(iterable, pd.DataFrame):
            iterable = [iterable]

        for i in iterable:
            print(i.to_csv(sep=","))

    return wrapper


def parse_dic(line, filter=None):
    if filter is None:
        filter = set()
    elif isinstance(filter, str):
        filter = set(filter.split(","))
    elif isinstance(filter, list):
        filter = set(filter)

    hit = False
    for f in filter:
        if f in line:
            hit = True
            break
    if not hit:
        return None

    ret = kv_pattern.findall(line)
    dic = {}
    for k, _, v in ret:
        try:
            dic[k] = int(v)
        except Exception:
            dic[k] = v.strip()
    return dic


def is_overlapping(squares: list[dict]):
    overlap_locs = []

    def decode_square(square: dict):
        if "tensor_size" in square and "end_addr" not in square:
            square["end_addr"] = square["start_addr"] + square["tensor_size"]
        try:
            return (
                square["start_addr"],
                square["end_addr"],
                square["live_start"],
                square["live_end"],
            )
        except:
            breakpoint()

    for i, j in combinations(range(len(squares)), 2):
        A, B = squares[i], squares[j]
        x1_A, x2_A, y1_A, y2_A = decode_square(A)
        x1_B, x2_B, y1_B, y2_B = decode_square(B)
        if max(x1_A, x1_B) < min(x2_A, x2_B) and max(y1_A, y1_B) < min(y2_A, y2_B):
            if A["op"] == "tpu.Reshape" or B["op"] == "tpu.Reshape":
                continue
            overlap_locs.append((squares[i], squares[j]))
            breakpoint()
    return overlap_locs


def detech_overlapping(log_file):
    """
    set_address,live_range,gmem_allocator,on_live_range
    """

    name_to_square = {}

    with Path(log_file).open("r") as r:
        for line in r:
            ret = parse_dic(line, {"setAddress", "assignGaddr", "update_live_range"})

            if ret is not None:
                name_to_square.setdefault(ret["loc"], {}).update(ret)

    ret = is_overlapping([i for i in name_to_square.values() if "live_start" in i])


def show_cut_results_costs(log_file):
    cut_ret = []
    cost_ret = []
    with Path(log_file).open("r") as r:
        for line in r:
            dic = parse_dic(line, "cut_optimize")
            if dic:
                if dic["step"] == "show_cut_results":
                    start_idx, end_idx = dic["range"].split("-")
                    dic["start_idx"] = int(start_idx)
                    dic["end_idx"] = int(end_idx)
                    cut_ret.append(dic)

            dic = parse_dic(line, "lg_cost")
            if dic:
                cost_ret.append(dic)

    cut_df = pd.DataFrame.from_records(cut_ret)
    cost_df = pd.DataFrame.from_records(cost_ret)

    cost_df = cost_df.set_index(["start_idx", "end_idx", "group_idx"])

    cut_ret.clear()
    for row in cut_df.to_dict("records"):
        # breakpoint()
        row["group_cost"] = cost_df.loc[(row["start_idx"], row["end_idx"], row["group_idx"])]["group_cost"].iloc[0]
        cut_ret.append(row)

    cut_df = pd.DataFrame.from_records(cut_ret)

    return cut_df
    # print(cost_df.to_csv(sep=","))


def cut_points(log_file):
    ret = []
    with Path(log_file).open("r") as r:
        for line in r:
            dic = parse_dic(line, "cut_points")
            if dic:
                ret.append(dic)

    df = pd.DataFrame.from_records(ret)

    for idx, sdf in df.groupby("group_idx"):
        sdf = sdf.drop_duplicates(["start", "end"], keep="last")
        min_idx = int(sdf["start"].min())
        max_idx = int(sdf["end"].max())
        sdf = sdf.set_index(["start", "end"])
        matrix = []
        for i in range(min_idx, max_idx + 1):
            row = []
            for j in range(min_idx, max_idx + 1):
                try:
                    value = sdf.loc[(i, j)]["cut_points"]
                    if value > 2**62:
                        row.append(None)
                    else:
                        row.append(value)
                except KeyError:
                    row.append(None)
            matrix.append(row)
        mdf = pd.DataFrame(matrix)
        yield mdf


def cost_table(log_file):
    """-debug-only=lg_cost"""

    ret = []
    with Path(log_file).open("r") as r:
        for line in r:
            dic = parse_dic(line, "lg_cost")
            if dic:
                ret.append(dic)

    df = pd.DataFrame.from_records(ret)

    for idx, sdf in df.groupby("group_idx"):
        sdf = sdf.drop_duplicates(["start_idx", "end_idx"], keep="first")
        min_idx = int(sdf["start_idx"].min())
        max_idx = int(sdf["end_idx"].max())
        sdf = sdf.set_index(["start_idx", "end_idx"])
        matrix = []
        for i in range(min_idx, max_idx + 1):
            row = []
            for j in range(min_idx, max_idx + 1):
                try:
                    value = sdf.loc[(i, j)]["group_cost"]
                    if value > 2**62:
                        row.append(None)
                    else:
                        row.append(value)
                except KeyError:
                    row.append(None)
            matrix.append(row)
        mdf = pd.DataFrame(matrix)

        print(mdf.to_csv(sep=","))
        yield mdf


def draw_live_range(log_file):
    dic = []
    with Path(log_file).open("r") as r:
        for line in r:
            ret = parse_dic(line, "live_range")
            if ret is not None:
                dic.append(ret)
    df = pd.DataFrame.from_records(dic)
    ratio = df["tensor_size"].max() // 1024
    breakpoint()


def grep_lg_index(log_file, start_idx, end_idx, group_idx=0):
    """
    grep and print log with start_idx and end_idx

    Example:
    >> logdebug_tool grep_lg_index log.txt 11 12 0 > log_11_12_0.txt
    >> ; action = lg_index; start_idx = 11; end_idx = 12; group_idx = 0
    >> ...
    >> ; action = lg_index; start_idx = x; end_idx = y; group_idx = 0

    DEBUG_TYPE: lg_index
    """
    log_file = Path(log_file)
    if not log_file.exists():
        print(f"Error: {log_file} does not exist")
        return

    first_lg_index = False
    lines = []
    with open(log_file, "r") as f:
        for line in f:
            if first_lg_index:
                lines.append(line)
            if "lg_index" not in line:
                continue

            dic = parse_dic(line)

            # breakpoint()
            if dic["start_idx"] == start_idx and dic["end_idx"] == end_idx:
                lines.append(line)
                first_lg_index = True
            elif first_lg_index:
                break

    print("".join(lines))


entry = {
    "grep_lg_index": grep_lg_index,
    "cut_points": comsume_in_main(cut_points),
    "cost_table": comsume_in_main(cost_table),
    "draw_live_range": draw_live_range,
    "show_cut_results_costs": comsume_in_main(show_cut_results_costs),
}


if __name__ == "__main__":
    fire.Fire(entry)
