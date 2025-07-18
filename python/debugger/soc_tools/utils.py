import argparse
import getpass
import os
import shutil
import time
import paramiko

clean_list = []


def parse_args():
    parser = argparse.ArgumentParser(description="Soc bmodel infer combine")
    parser.add_argument(
        "--path",
        default="/tmp",
        help="The folder should contain the BModel, its input_data, and reference files.",
    )
    parser.add_argument(
        "--bmodel",
        required=True,
        help="Bmodel file path",
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Input file path",
    )
    parser.add_argument(
        "--ref",
        required=True,
        help="Ref file path, only support .npz",
    )
    parser.add_argument(
        "--tool_path",
        default="/tmp",
        help="The folder where place the soc_infer dir.",
    )
    parser.add_argument(
        "--out_fixed",
        action="store_true",
        help="Whether to output fixed number.",
    )
    parser.add_argument("--enable_log", action="store_true", help="Whether to enable log.")
    parser.add_argument(
        "--using_memory_opt",
        action="store_true",
        help="Whether to enable memory opt, which decrease memory usage but increase time cost.",
    )
    parser.add_argument(
        "--run_by_atomic",
        action="store_true",
        help="Whether to run by atomic cmds, instead of running by ops as default.",
    )
    parser.add_argument(
        "--desire_op",
        type=str,
        default="",
        help="Whether to only dump specific ops, dump all ops as default.",
    )
    args = parser.parse_args()
    return args


def _soc_upload_dir(sftp, local_dir, remote_dir):
    sftp.mkdir(remote_dir)
    for root, dirs, files in os.walk(local_dir):
        for file in files:
            local_path = os.path.join(root, file)
            remote_path = os.path.join(remote_dir, os.path.relpath(local_path, local_dir))
            sftp.put(local_path, remote_path)

        for dir in dirs:
            local_path = os.path.join(root, dir)
            remote_path = os.path.join(remote_dir, os.path.relpath(local_path, local_dir))
            try:
                sftp.mkdir(remote_path)
            except:
                pass


def _soc_mk_dir(client, remote_dir):
    command = f"mkdir -p {remote_dir}"
    stdin, stdout, stderr = client.exec_command(command)
    if stderr:
        print(stderr.read().decode("utf-8"))


def _soc_rm_dir(client, remote_dir):
    command = f"rm -rf {remote_dir}"
    stdin, stdout, stderr = client.exec_command(command)
    if stderr:
        print(stderr.read().decode("utf-8"))


def progress(size, total):
    progress = (size / total) * 100
    print(f"Processing : {size}/{total} bytes ({progress:.2f}%)\r", end="")


def progress_put(sftp, file_name, local_path="", remote_path="", progress=None):
    print(f"Transfering {file_name}:")
    sftp.put(local_path, remote_path, callback=progress)
    print()


def progress_get(sftp, file_name, local_path="", remote_path="", progress=None):
    print(f"Getting {file_name}:")
    sftp.get(remote_path, local_path, callback=progress)
    print()


def add_to_clean(path):
    clean_list.append(path)


def clean_collected_files():
    for item in clean_list:
        if os.path.exists(item):
            if os.path.isfile(item):
                print(f"Cleaning file: {item}")
                os.remove(item)
            elif os.path.isdir(item):
                print(f"Cleaning directory: {item}")
                shutil.rmtree(item)
            else:
                print(f"Skipping unknown type: {item}")
        else:
            print(f"Item not found: {item}")


# def collect_copy(src_file, dst_path):
#     file_name = os.path.basename(src_file)
#     add_to_clean(os.path.join(dst_path, file_name))
#     shutil.copy(src_file, dst_path)


def collect_copytree(src_path, dst_path):
    dir_name = os.path.basename(src_path)
    # add_to_clean(os.path.join(dst_path, dir_name))
    shutil.copytree(src_path, os.path.join(dst_path, dir_name), dirs_exist_ok=True)


def collect_files():
    cur_dir = os.path.dirname(__file__)
    soc_tool_dir = os.path.join(cur_dir, "debugger")
    add_to_clean(soc_tool_dir)
    os.makedirs(os.path.join(soc_tool_dir, "lib"), exist_ok=True)
    utils_dir = os.path.join(cur_dir, "../../utils")
    thirdparty_lib_dir = os.path.join(cur_dir, "../../../third_party")
    shutil.copy(os.path.join(cur_dir, "../disassembler.py"), soc_tool_dir)
    shutil.copy(os.path.join(cur_dir, "../final_mlir.py"), soc_tool_dir)
    shutil.copy(os.path.join(cur_dir, "soc_atomic_dialect.py"),
                os.path.join(soc_tool_dir, "atomic_dialect.py"))
    shutil.copy(os.path.join(utils_dir, "lowering.py"), soc_tool_dir)
    shutil.copy(os.path.join(thirdparty_lib_dir, "atomic_exec/libatomic_exec_aarch64.so"),
                os.path.join(soc_tool_dir, "lib"))
    shutil.copy(os.path.join(thirdparty_lib_dir, "atomic_exec/libatomic_exec_bm1688_aarch64.so"),
                os.path.join(soc_tool_dir, "lib"))
    shutil.copy(
        os.path.join(thirdparty_lib_dir, "atomic_exec/libbm1684x_atomic_kernel.so"),
        os.path.join(soc_tool_dir,
                     "lib"))  # do not use libbm1684x_kernel_module, which may cause nan error
    shutil.copy(os.path.join(thirdparty_lib_dir, "nntoolchain/lib/libbmtpulv60_kernel_module.so"),
                os.path.join(soc_tool_dir, "lib"))

    local_tools_path = os.getenv("PROJECT_ROOT", None)
    if not local_tools_path:
        local_tools_path = os.getenv("TPUC_ROOT")
        assert local_tools_path
    if (os.path.exists(os.path.join(local_tools_path, "./install/python/debugger/bmodel_fbs.py"))):
        shutil.copy(os.path.join(local_tools_path, "./install/python/debugger/bmodel_fbs.py"),
                    soc_tool_dir)
    elif (os.path.exists(os.path.join(local_tools_path, "./python/debugger/bmodel_fbs.py"))):
        shutil.copy(os.path.join(local_tools_path, "./python/debugger/bmodel_fbs.py"), soc_tool_dir)

    collect_copytree(os.path.join(cur_dir, "../target_common"), soc_tool_dir)
    collect_copytree(os.path.join(cur_dir, "../target_1684x"), soc_tool_dir)
    collect_copytree(os.path.join(cur_dir, "../target_1688"), soc_tool_dir)
    collect_copytree(os.path.join(cur_dir, "../target_1684"), soc_tool_dir)
    collect_copytree(os.path.join(cur_dir, "../target_1690"), soc_tool_dir)
    collect_copytree(os.path.join(cur_dir, "../target_2380"), soc_tool_dir)
    collect_copytree(os.path.join(cur_dir, "../target_cv184x"), soc_tool_dir)
    collect_copytree(os.path.join(cur_dir, "../target_sgtpuv8"), soc_tool_dir)


# connect remote ssh server
def soc_connect(hostname, port, username, password):
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    if not hostname:
        hostname = input("Enter the hostname: ")
    if not port:
        port = int(input("Enter the port: "))
    if not username:
        username = input("Enter your username: ")
    if not password:
        password = getpass.getpass("Enter your password: ")

    client.connect(hostname=hostname, port=port, username=username, password=password)
    sftp = paramiko.SFTPClient.from_transport(client.get_transport())
    return client, sftp


# transfer cache files and soc_infer tools
def soc_trans_files(client, sftp, local_path, remote_path, bmodel, input, ref):
    # transfer file
    _soc_mk_dir(client, remote_path)
    sftp.chdir(remote_path)
    progress_put(sftp, "cmds.pkl", os.path.join(local_path, "soc_tmp", "cmds.pkl"), "cmds.pkl",
                 progress)
    progress_put(sftp, "values_in.pkl", os.path.join(local_path, "soc_tmp", "values_in.pkl"),
                 "values_in.pkl", progress)
    progress_put(sftp, "values_out.pkl", os.path.join(local_path, "soc_tmp", "values_out.pkl"),
                 "values_out.pkl", progress)
    progress_put(sftp, "bmodel file", bmodel, os.path.basename(bmodel), progress)
    progress_put(sftp, "input data", input, os.path.basename(input), progress)
    progress_put(sftp, "ref data", ref, os.path.basename(ref), progress)

    # transfer soc_infer tools
    print("Transfering Soc_infer Tools...")
    local_tools_path = os.getenv("PROJECT_ROOT", None)
    if not local_tools_path:
        local_tools_path = os.getenv("TPUC_ROOT")
        assert local_tools_path
    _soc_upload_dir(sftp, os.path.join(local_tools_path, "python/debugger/soc_tools/"),
                    os.path.join(remote_path, "soc_tools"))


def soc_check_end_status(exec_command, client, sftp, soc_tool_path):
    print(f"soc execute command: {exec_command}")
    print("####### REMOTE OUTPUTS START #######\n")

    shell = client.invoke_shell()
    shell.send(exec_command + "\n")
    while True:
        try:
            sftp.stat(os.path.join(soc_tool_path, "log.txt"))
            print("REMOTE PROGRESS FINISHED!")
            break
        except FileNotFoundError:
            time.sleep(0.5)  # refresh output every 0.5s
            if shell.recv_ready():
                output = shell.recv(1024).decode("utf-8", errors="ignore")
                print(output, end="")
            if shell.recv_stderr_ready():
                error = shell.recv_stderr(1024).decode("utf-8", errors="ignore")
                print(error, end="")
                print("Error encountered, exiting.")
                shell.close()
                return
    shell.close()
    print("######## REMOTE OUTPUTS END ########\n")


def soc_fetch_log_and_npz(client, sftp, local_path, remote_path, remote_ref, enable_soc_log=False):
    if enable_soc_log:
        progress_get(
            sftp,
            "stdout file",
            os.path.join(local_path, "log.txt"),
            os.path.join(remote_path, "log.txt"),
            progress,
        )
        progress_get(
            sftp,
            "stderr file",
            os.path.join(local_path, "nohup.out"),
            os.path.join(remote_path, "nohup.out"),
            progress,
        )
        print(
            f"log file recieved at {os.path.join(local_path, 'log.txt')} and {os.path.join(local_path, 'nohup.out')}"
        )

    # retrieve results
    remote_infer_combine_path = os.path.join(remote_path, f"soc_infer_{remote_ref}")
    local_infer_combine_path = os.path.join(local_path, f"soc_infer_{remote_ref}")
    progress_get(
        sftp,
        "Inference_combine file",
        local_infer_combine_path,
        remote_infer_combine_path,
        progress,
    )


if __name__ == "__main__":
    collect_files("1684x")
    clean_collected_files()
