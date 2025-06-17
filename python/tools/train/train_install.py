import sys, os, subprocess, pkg_resources

package_name = "tpu_mlir"

# get tpu_mlir package path
# try:
#     distribution = pkg_resources.get_distribution(package_name)
#     package_path = distribution.location + f"/{package_name}"
# except pkg_resources.DistributionNotFound:
#     raise RuntimeError(f"Package '{package_name}' is not installed. Exiting.")

package_path = os.path.dirname(os.path.abspath(__file__))

# set execute permissions
permission_tag = os.stat(f"{package_path}/python/tools/model_transform.py").st_mode
tools_path = f"{package_path}/python/tools"
if not (permission_tag & os.X_OK):
    os.system(f"chmod +x {tools_path}/*")
    print(f"Execute permissions added to all files in '{tools_path}'.")

# env path init
new_path = [
    f"{package_path}/bin:",
    f"{package_path}/python/tools:",
    f"{package_path}/python/utils:",
    f"{package_path}/python/test:",
    f"{package_path}/python/samples:",
    f"{package_path}/customlayer/python:",
]
os.environ["PATH"] += "".join(new_path)
os.environ["PYTHONPATH"] = (f"{package_path}/:" + f"{package_path}/python/:" +
                            f"{package_path}/regression/:" + f"{package_path}/customlayer/python/:")
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["TPUC_ROOT"] = f"{package_path}"
os.environ["LD_LIBRARY_PATH"] = (f"{package_path}/lib:" + f"{package_path}/lib/capi:" +
                                 f"{package_path}/lib/third_party:")
os.environ["TPUKERNEL_CUSTOM_FIRMWARE_PATH"] = f"{package_path}/lib/libcmodel_custom.so"
os.environ["CUSTOM_LAYER_UNITTEST_DIR"] = f"{package_path}/customlayer/test_if/unittest"

# set python package searching path
sys.path.append(f"{package_path}/")
sys.path.append(f"{package_path}/python/")
sys.path.append(f"{package_path}/regression/")
sys.path.append(f"{package_path}/customlayer/python/")


def run_subprocess_py(file_name):
    arguments = sys.argv[1:]
    if arguments:
        command = ["python3", file_name] + arguments
    else:
        command = ["python3", file_name]
    process = subprocess.Popen(command)
    return_code = process.wait()
    if return_code != 0:
        exit(1)


def run_subprocess_c(file_name):
    arguments = sys.argv[1:]
    if arguments:
        command = [file_name] + arguments
    else:
        command = [file_name]
    process = subprocess.Popen(command)
    return_code = process.wait()
    if return_code != 0:
        exit(1)


# def cp_from_package_root():
#     arguments = sys.argv[1:]
#     if len(arguments) != 2:
#         print("Both src_path and target_path are and only required!")
#         exit(1)
#     command = ["cp", "-rf"] + [f"{package_path}/" + arguments[0]] + [arguments[1]]
#     process = subprocess.Popen(command)
#     return_code = process.wait()
#     if return_code != 0:
#         exit(1)
