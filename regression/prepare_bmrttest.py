#!/usr/bin/env python3
import subprocess
import re
import os
import numpy as np
import sys


def get_bmodel_io_info(bmodel):
    input_rg = "^input: .+"
    output_rg = "^output: .+"
    shape_rg = "\[((\d+),\ *)*\d+\]"

    def name_shape(info):
        shape = re.search(shape_rg, info.group(0)).group(0)  # type: ignore
        name = info.group(0).split(",")[0].split(":")[1].strip()
        return name, shape

    inputs = []
    outputs = []
    try:
        out = subprocess.run(
            ["model_tool", "--info", bmodel],
            capture_output=True,
            check=True,
        )
        for meg in out.stdout.decode().split("\n"):
            m = re.search(input_rg + shape_rg, meg)
            if m:
                inputs.append(name_shape(m)[0])
                continue
            m = re.search(output_rg + shape_rg, meg)
            if m:
                outputs.append(name_shape(m)[0])

    except subprocess.CalledProcessError:
        print(f"run 'model_tool --info {bmodel}' failed.")
        return [], []
    return (inputs, outputs)


def get_io_data(dir):

    __regx = dict(
        indata=re.compile(".+_in_f32\.npz$"),
        f32_bmodel=re.compile(".+f32\.bmodel$"),
        int8_asym_bmodel=re.compile(".+int8_asym\.bmodel$"),
        int8_sym_bmodel=re.compile(".+int8_sym\.bmodel$"),
        int8_asym_outdata=re.compile(".+int8_asym_model_outputs\.npz$"),
        int8_sym_outdata=re.compile(".+int8_sym_model_outputs\.npz$"),
        f32_outdata=re.compile(".+f32_model_outputs\.npz$"),
    )
    files = os.listdir(dir)
    name = os.path.basename(dir)
    out = {}
    for k, v in __regx.items():
        for f in files:
            m = v.search(f)
            if m:
                out[k] = m.group(0)
        if k not in out:
            out[k] = None
    return name, out


def prepare_bmodel_test(dir):
    name, files = get_io_data(dir)
    f32_bmodel = files["f32_bmodel"]
    int8_asym_bmodel = files["int8_asym_bmodel"]
    int8_sym_bmodel = files["int8_sym_bmodel"]

    def prepare_test(name, bmodel, indata, outdata):
        bmodel = os.path.join(dir, bmodel)
        indata = os.path.join(dir, indata)
        outdata = os.path.join(dir, outdata)
        in_name, out_name = get_bmodel_io_info(bmodel)
        if in_name == []:
            Warning(f"get {bmodel} information failed.")
        os.makedirs(name, exist_ok=True)
        os.rename(bmodel, os.path.join(name, "compilation.bmodel"))
        profile_log = bmodel + ".compiler_profile_0.txt"
        os.rename(profile_log, os.path.join(name, "compiler_profile_0.txt"))
        indata = np.load(indata)
        with open(name + "/input_ref_data.dat", "wb") as f:
            for i in in_name:
                indata[i].tofile(f)
        outdata = np.load(outdata)
        with open(name + "/output_ref_data.dat", "wb") as f:
            for o in out_name:
                outdata[o].tofile(f)

    if f32_bmodel:
        prepare_test(name + "_f32", f32_bmodel, files["indata"], files["f32_outdata"])
    if int8_sym_bmodel:
        prepare_test(
            name + "_int8_sym",
            int8_sym_bmodel,
            files["indata"],
            files["int8_sym_outdata"],
        )
    if int8_asym_bmodel:
        prepare_test(
            name + "_int8_asym",
            int8_asym_bmodel,
            files["indata"],
            files["int8_asym_outdata"],
        )


def bmodel_out(dir):
    folders = os.listdir(dir)
    except_folder = ["step_by_step", "onnx_test"]
    folders = list(set(folders).difference(set(except_folder)))
    for f in folders:
        folder = os.path.join(dir, f)
        prepare_bmodel_test(folder)


if __name__ == "__main__":
    args = sys.argv
    assert (
        len(args) == 2
    ), f"The input should be a folder. but more arguments are provided {args}"
    bmodel_out(args[1])
