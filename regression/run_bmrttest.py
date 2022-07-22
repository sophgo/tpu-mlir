#!/usr/bin/env python3
from time import gmtime, strftime
import subprocess
import os, sys, re
from collections import OrderedDict


def get_bmrtt_log(bmodel, loopnum=100):
    status_common = ".+The\ network\[.+\]\ stage\[\d+\]\ cmp\ {} \+{{3}}"
    failed_regx = re.compile(status_common.format("failed"))
    success_regx = re.compile(status_common.format("success"))
    time_common = "INFO:{}\ +time\(s\):\ +\d+\.\d+"
    time_us = re.compile("\w+\ us")
    shape_rg = "\[( *\d+ *)*\]"
    input_regx = re.compile("Input\ +\d+\).+shape=" + shape_rg)
    shape_regx = re.compile(shape_rg)

    def get_time(msg):
        return float(msg.split(":")[-1])

    def get_times(msg):
        time = [float(x.split(" ")[0]) for x in time_us.findall(msg)]
        return dict(zip(("_total (us)", "_npu (us)", "_cpu (us)"), time))

    time_regx = {
        "compute": (
            re.compile(
                "INFO:net\[.+\]\ stage\[\d+\],\ launch\ total\ time\ is\ \d+\ us.+"
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
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    for meg in out.stdout.decode().split("\n"):
        if input_regx.search(meg):
            input = re.findall("\d+", shape_regx.search(meg).group())  # type:ignore
            out_msg.setdefault("shape", []).append(f"<{'x'.join(input)}>")
            continue
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
                break

    out = {}
    for k, v in out_msg.items():
        if isinstance(v[0], float):
            mean = sum(v) / len(v)
            out[k] = (mean, max(v) - mean)
        elif isinstance(v[0], str):  # input shape
            out[k] = v
        else:
            out[k] = f"{sum(v) / len(v):.2%}"
    out["loopnum"] = loopnum
    out["bmodel"] = os.path.basename(bmodel)
    return out


# from tpu-perf
def parse_profile(folder):
    with open(os.path.join(folder, "compiler_profile_0.txt")) as f:
        lines = f.read()
    lines = lines[lines.find("API_END") :]
    data = dict()
    for pair in re.finditer("(\w+) *: *((\d+\.)?\d+)", lines):
        v = pair.group(2)
        data[pair.group(1)] = float(v)

    return data


def get_profile(path):
    bmrt_log = get_bmrtt_log(path)
    profile_log = parse_profile(path)
    time = [x * 1000 for x in bmrt_log["calculate (s)"]]  # seconds to milliseconds
    mac_util = lambda t: 100 * profile_log["flops"] / (t * 1e-3 * 1e12 * 32)
    return OrderedDict(
        {
            "name": bmrt_log["bmodel"],
            "shape": " ".join(bmrt_log["shape"]),
            "time": f"{time[0]:.3f} +/-{time[1]:.3f}",
            "cmodel_estimate_time": f"{profile_log['runtime']:.3f}",
            "mac_utilization": f"{mac_util(time[0]):.2f}",
            "cmodel_estimate_mac_utilization": (
                f"{mac_util(profile_log['ComputationAbility']):.2f}"
            ),
            "ddr_utilization": f"{profile_log['USAGE']:.2f}",
            "cmodel_estimate_ddr_bandwidth": (
                f"{profile_log['USAGE'] * profile_log['runtime'] / time[0]:.2f}"
            ),
        }
    )


def gen_bmodel_test_info(dir):
    files = os.listdir(dir)
    fname = f"{strftime('%Y-%m-%d_%H.%M.%S.csv', gmtime())}"
    write_header = True
    with open(fname, "w") as fb:
        for f in files:
            folder = os.path.join(dir, f)
            if os.path.isdir(folder):
                info = get_profile(os.path.join(dir, f))
                if write_header:
                    fb.write(f"{list(info.keys())}"[1:-1] + "\n")
                    write_header = False
                fb.write(", ".join(info.values()) + "\n")


if __name__ == "__main__":
    args = sys.argv
    assert (
        len(args) <= 2
    ), f"The input should be a folder. but more arguments are provided {args}"
    if len(args) == 2:
        gen_bmodel_test_info(args[1])
    elif len(args) == 1:
        gen_bmodel_test_info("./")
