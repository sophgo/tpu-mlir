#!/usr/bin/python
"""
# 多次运行bmodel，获取耗时相关统计值：均值、最小值、最大值、标准差
$ remote_test.py --username xxx --hostname ip_addr --bmodel ./xxx.bmodel --password xxx --debug statistic

# 获取profile文件
$ remote_test.py --username xxx --hostname ip_addr --bmodel ./xxx.bmodel --password xxx --debug profile

# 如果远程已存在该bmodel，--rename会给远程文件名添加后缀，否则会覆盖
$ remote_test.py --username xxx --hostname ip_addr --bmodel ./xxx.bmodel --password xxx --rename
"""
import os
import time
import getpass
import paramiko
import argparse
# import yaml
import sys
from tqdm import tqdm
import re
import numpy as np

# def parse_config(config_file):
#     with open(config_file, 'r') as f:
#         data = yaml.safe_load(f)
#     return data

def progress_bar(finished, total):
    """显示上传/下载进度的函数"""
    done = int(50 * finished / total)
    sys.stdout.write("\r[{}{}] {:.2f}%".format(
        '█' * done,
        '.' * (50-done),
        float(finished)/total*100
    ))
    if finished >= total:
        sys.stdout.write('\n')
    sys.stdout.flush()

def analysis_time(log_content):
    matches = re.findall(r'launch total time is (\d+) us', log_content)
    time_us_list = [float(t) for t in matches]
    time_us = np.array(time_us_list)
    time_mean = time_us.mean()
    time_min = time_us.min()
    time_max = time_us.max()
    time_std = time_us.std()
    print(f'total_time statistic: mean={time_mean:.2f} us, min={time_min:.2f} us, max={time_max:.2f} us, std={time_std:.2f} us\n')

# class LocalRunner:
#     def __init__(self, opt):
#         pass

class RemoteRunner:
    def __init__(self, opt):
        self.hostname = opt.hostname
        self.username = opt.username
        self.password = opt.password
        self.local_path = os.path.dirname(opt.bmodel)
        self.local_file = opt.bmodel
        self.bmodel_name = os.path.basename(opt.bmodel)
        self.remote_path = opt.remote_path
        self.remote_file = os.path.join(self.remote_path, self.bmodel_name)
        self.ssh_client = None
        self.sftp_client = None
        self.opt = opt

        if not self.password:
            self.password = getpass.getpass("Please input your password: ")
        self.connect()

    def connect(self):
        self.ssh_client = paramiko.SSHClient()
        self.ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy)

        try:
            self.ssh_client.connect(self.hostname, username=self.username, password=self.password)
        except Exception as e:
            print(f"Failed to connect: {e}")
            self.ssh_client.close()
            self.ssh_client = None

        self.sftp_client = self.ssh_client.open_sftp()

    def run_command(self):
        if not self.ssh_client:
            print("Not connected, Please connect first.")
            return None
        remote_file = os.path.join(self.remote_path, self.bmodel_name)
        pre_cmd, cmd, post_cmd = self.get_bmrt_cmd(remote_file, self.opt.debug)
        if pre_cmd:
            print(pre_cmd)
            stdin, stdout, stderr = self.ssh_client.exec_command(pre_cmd)
            result = stdout.read().decode('utf-8')
            print(result)

        if cmd:
            print(cmd)
            stdin, stdout, stderr = self.ssh_client.exec_command(cmd)
            result = stdout.read().decode('utf-8')
            print(result)
            if self.opt.debug == 'statistic':
                analysis_time(result)

        if post_cmd:
            print(post_cmd)
            stdin, stdout, stderr = self.ssh_client.exec_command(post_cmd)
            result = stdout.read().decode('utf-8')
            print(result)

    def close_connection(self):
        if self.ssh_client:
            self.ssh_client.close()

    def check_create_remote_path(self):
        cmd = f'if [ ! -d {self.remote_path} ]; then mkdir -p {self.remote_path} && echo "mkdir {self.remote_path}"; fi'
        stdin, stdout, stderr = self.ssh_client.exec_command(cmd)
        result = stdout.read().decode('utf-8')
        print(result)

    def put_file(self):
        self.check_create_remote_path()
        suffix = 1
        while True:
            try:
                self.sftp_client.stat(self.remote_file)
                if self.opt.rename:
                    name, ext = os.path.splitext(self.bmodel_name)
                    self.remote_file = os.path.join(self.remote_path, f'{name}_{suffix}{ext}')
                    suffix += 1
                else:
                    print(f"Skip sending file because {self.bmodel_name} has existed.")
                    return
            except IOError as e:
                break
        file_size = os.path.getsize(self.opt.bmodel)
        print(f"== Start to send {self.local_file} to remote server.")
        self.sftp_client.put(self.local_file, self.remote_file,
                            callback=lambda x, y: progress_bar(x, file_size))
        print(f"== Finished.")

    def get_file(self):
        if not self.opt.debug:
            return

        local_path = os.path.dirname(self.opt.bmodel)
        filename = 'bmprofile_data-1.tar.gz'
        local_file = os.path.join(local_path, filename)

        remote_file = os.path.join(self.opt.remote_path, filename)
        file_size = self.sftp_client.stat(remote_file).st_size
        print(f"== Start to receive {remote_file} from remote server.")
        self.sftp_client.get(remote_file, local_file,
                            callback=lambda x, y: progress_bar(x, file_size))
        print(f"== Finished.")


    def get_bmrt_cmd(self, remote_bmodel_file, debug='profile'):
        remote_path = os.path.dirname(remote_bmodel_file)

        pre_remote_cmd = None
        remote_cmd = '/opt/sophon/libsophon-current/bin/bmrt_test --bmodel ' + remote_bmodel_file
        post_remote_cmd = None
        if debug == 'profile':
            pre_remote_cmd = f'cd {remote_path} && rm -rf ./bmprofile*'
            remote_cmd = 'BMRUNTIME_ENABLE_PROFILE=1 ' + remote_cmd
            post_remote_cmd = f'cd {remote_path} && tar -czf bmprofile_data-1.tar.gz bmprofile_data-1/'
        elif debug == 'statistic':
            remote_cmd = remote_cmd + f' --loopnum 30'
        remote_cmd = f'cd {remote_path} && ' + remote_cmd

        return pre_remote_cmd, remote_cmd, post_remote_cmd


def parse_opt():
    parser = argparse.ArgumentParser()
    # yapf: disable
    parser.add_argument(
        "--bmodel",
        type=str,
        help="The bmodel that will be passed to the remote server")
    parser.add_argument(
        "--username",
        type=str,
        default="user",
        help="The username to launch the server.")
    parser.add_argument(
        "--hostname",
        type=str,
        default="local",
        help="The server IP address.")
    parser.add_argument(
        "--password",
        type=str,
        default="",
        help="the password for hostname")
    parser.add_argument(
        "--remote_path",
        type=str,
        help="The remote path to store the bmodel. It should be absolute path.",
        default="/tmp")
    parser.add_argument(
        "--rename",
        action="store_true",
        help="rename the bmodel file if file exists in remote_path")
    parser.add_argument(
        "--ext_args",
        type=str,
        default="",
        help="the extra args for bmrt_test, such as '--devid 4 --loopnum 100'")
    parser.add_argument(
        "--debug",
        type=str.lower,
        choices=['profile', 'statistic'],
        help="profile: generate profile file. statistic: get mean/min/max/std of total time")
    parser.add_argument(
        "--config_file",
        type=str,
        default="",
        help="The config file path")
    # yapf: enable
    opt = parser.parse_args()
    return opt

def main(opt):
    runner = RemoteRunner(opt)
    runner.put_file()
    runner.run_command()
    runner.get_file()
    runner.close_connection()

if __name__ == '__main__':
    opt = parse_opt()
    main(opt)
