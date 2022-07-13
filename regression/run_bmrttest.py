#!/usr/bin/env python3
from time import gmtime, strftime
import subprocess
import numpy as np
import os
import sys
import re


def run_bmrtttest(bmodel, loopnum=100):
    status_common = (
        "\[BMRT\]\[bmrt_test:\d+].+The\ network\[.+\]\ stage\[\d+\]\ cmp\ {} \+{{3}}"
    )
    failed_regx = re.compile(status_common.format("failed"))
    success_regx = re.compile(status_common.format("success"))
    time_common = "\[BMRT\]\[bmrt_test:\d+]\ INFO:{}\ +time\(s\):\ +\d+\.\d+"
    time_us = re.compile("\w+\ us")

    def get_time(msg):
        return float(msg.split(":")[-1])

    def get_times(msg):
        time = [float(x.split(" ")[0]) for x in time_us.findall(msg)]
        return dict(zip(("_total (us)", "_npu (us)", "_cpu (us)"), time))

    time_regx = {
        "compute": (
            re.compile(
                "\[BMRT\]\[bmrt_test:\d+\]\ INFO:net\[.+\]\ stage\[\d+\],\ launch\ total\ time\ is\ \d+\ us.+"
            ),
            get_times,
        ),
        "load_input (s)": (re.compile(time_common.format("load\ input")), get_time),
        "calculate (s)": (re.compile(time_common.format("calculate")), get_time),
        "get_output (s)": (re.compile(time_common.format("get\ output")), get_time),
    }
    out_msg = {}

    print(f"[run test]: {bmodel}")
    out = subprocess.run(
        ["bmrt_test", "--context_dir", bmodel, "--loopnum", str(loopnum)],
        capture_output=True,
    )
    for meg in out.stdout.decode().split("\n"):
        if failed_regx.search(meg):
            out_msg.setdefault("success", []).append(False)
            continue
        if success_regx.search(meg):
            out_msg.setdefault("success", []).append(True)
            continue
        for k, r in time_regx.items():
            __r, f = r
            if __r.search(meg):
                t = f(meg)
                if isinstance(t, dict):
                    for _k, _t in t.items():
                        out_msg.setdefault(k + _k, []).append(_t)
                else:
                    out_msg.setdefault(k, []).append(f(meg))
                continue

    out = {}
    for k, v in out_msg.items():
        if isinstance(v[0], float):
            __out = (np.min(v), np.max(v), np.mean(v))
            out[k + " min"] = __out[0]
            out[k + " max"] = __out[1]
            out[k + " mean"] = __out[2]
        else:
            out[k] = f"{sum(v) / len(v):.2%}"
    out["loopnum"] = loopnum
    out["bmodel"] = os.path.basename(bmodel)
    return out


def gen_bmodel_test_info(dir):
    files = os.listdir(dir)
    bmrt_test_info = []
    keys = set()
    for f in files:
        info = run_bmrtttest(os.path.join(dir, f))
        keys.update(info.keys())
        bmrt_test_info.append(info)
    keys = sorted(keys)
    fname = f"{strftime('%Y-%m-%d_%H.%M.%S.csv', gmtime())}"
    with open(fname, "w") as fb:
        fb.write(f"{keys}"[1:-1] + "\n")
        for info in bmrt_test_info:
            _placeholder = dict.fromkeys(keys)
            _placeholder.update(info)
            fb.write(", ".join([str(_placeholder[k]) for k in keys]) + "\n")


if __name__ == "__main__":
    args = sys.argv
    assert (
        len(args) == 2
    ), f"The input should be a folder. but more arguments are provided {args}"
    gen_bmodel_test_info(args[1])
