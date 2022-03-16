import subprocess


def checkReturnValue(ret, func: str):
    if ret.returncode == 0:
        print("{} run success".format(func))
    else:
        print("[!Error]cmd: {}".format(" ".join(ret.args)))
        print("error occured: {}, func: {}\nmsg: {}".format(ret.returncode, func, ret))


def mlir_opt(mlirfile, opt_mlirfile):
    ret = subprocess.run(["sophgo-opt", "--canonicalize", "--save-weight", mlirfile, "-o", opt_mlirfile])
    checkReturnValue(ret, "sophgo-opt")
    return ret.returncode


def f32_blobs_compare(a_npz: str, b_npz: str, tolerance: str, excepts=None, show_detail=True):
    cmd = ["npz_tool.py", "compare", a_npz, b_npz, "--tolerance", tolerance]
    if excepts:
        cmd.extend(["--except", excepts])
    if show_detail:
        cmd.append('-vv')
    ret = subprocess.run(cmd)
    checkReturnValue(ret, "compare")
    return ret.returncode
