# ==============================================================================
#
# Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
#
# TPU-MLIR is licensed under the 2-Clause BSD License except for the
# third-party components.
#
# ==============================================================================

import xmlrpc.client
import hashlib
import os


class file_helper:
    def __init__(self, file) -> None:
        if os.path.isfile(file):
            self.type = "file"
        elif os.path.isdir(file):
            self.type = "dir"
        else:
            raise Exception(f"target '{file}' is unavailable.")

        self.handle = open(file, "rb")
        self.md5 = hashlib.md5(self.handle.read()).hexdigest()
        self.name = file

    def get_buf(self):
        self.handle.seek(0)
        return (
            self.md5,
            self.name,
            xmlrpc.client.Binary(self.handle.read()),
        )


class SOCClient:
    def __init__(self, url="http://localhost:8000/") -> None:
        self.proxy = xmlrpc.client.ServerProxy(url, allow_none=True)

    def __send_file(self, file):
        try:
            file_des = file_helper(file)
            md5 = file_des.md5
            if not self.proxy.has_file(md5):
                print(f"sending: {os.path.basename(file)}")
                self.proxy.send_file(*file_des.get_buf())
                print("    >> finish.")
            return md5
        except Exception as e:
            raise e

    def __send_folder(self, folder):
        md5s = {}
        for f in os.listdir(folder):
            file = os.path.join(folder, f)
            if os.path.isfile(file):
                try:
                    md5s[f] = self.__send_file(file)
                except Exception as e:
                    raise e
        return self.proxy.build_dir(md5s)

    def send(self, item):
        if os.path.isdir(item):
            return self.__send_folder(item)
        if os.path.isfile(item):
            return self.__send_file(item)
        raise Exception(f"target '{item}' is unavailable.")

    def run_file(self, cmd, file):
        try:
            md5 = self.send(file) if file else None
            return self.proxy.run(cmd, md5)
        except Exception as e:
            raise e
