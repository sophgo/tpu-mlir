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

kv_pattern = re.compile(r";\s*([\w/.]+)\s*=\s*(['\"]?)([\w\s/\".]+)\2")


def parse_dic(line):
    ret = kv_pattern.findall(line)
    dic = {}
    for k, _, v in ret:
        try:
            dic[k] = int(v)
        except Exception:
            dic[k] = v
    return dic


class Entry:

    def grep_lg_index(self, log_file, start_idx, end_idx, group_idx=0):
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


if __name__ == "__main__":
    fire.Fire(Entry())
