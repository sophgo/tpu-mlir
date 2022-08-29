#!/usr/bin/env python3
# ==============================================================================
#
# Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
#
# TPU-MLIR is licensed under the 2-Clause BSD License except for the
# third-party components.
#
# ==============================================================================

from xmlrpc.server import SimpleXMLRPCServer
import subprocess
import hashlib
import os
import importlib
import pprint
import atexit
import shutil
import datetime
import argparse


def check_health(md5, buffer):
    return md5 == hashlib.md5(buffer.data).hexdigest()


def now():
    return str(datetime.datetime.now())


def time_obj(time_str):
    return datetime.datetime.strptime(time_str, "%Y-%m-%d %H:%M:%S.%f")


class FileRecorder:
    _folder = "./.cache/soc_rpc/"
    _recorder = os.path.join(_folder, "recorder.py")
    _max_capacity = 2**31
    _age = 2.0**1023
    __slots__ = ("record", "size")
    # {md5, {name, date, size, query}}
    def __init__(self) -> None:
        if os.path.exists(self._recorder):
            loader = importlib.machinery.SourceFileLoader("record", self._recorder)  # type: ignore
            spec = importlib.util.spec_from_loader("record", loader)  # type: ignore
            recorder = importlib.util.module_from_spec(spec)  # type: ignore
            loader.exec_module(recorder)
            self.record = recorder.record
        else:
            os.makedirs(os.path.dirname(self._recorder), exist_ok=True)
            self.record = {}
        self.size = sum((x["size"] for x in self.record.values()))
        self.eviction()

    def __contains__(self, key):
        if key in self.record:
            value = self.record[key]
            value["query"] += 1
            value["date"] = now()
            self.__aging()
            value["age"] += self._age
            return True
        return False

    def __getitem__(self, md5):
        return os.path.join(self._folder, md5)

    def __setitem__(self, md5, file):
        name, buffer = file
        assert check_health(md5, buffer), f"md5 does not match, file corrupt."
        name = os.path.basename(name)
        size = len(buffer.data)
        self.size += size
        with open(os.path.join(self._folder, md5), "wb") as handle:
            handle.write(buffer.data)
        self.record[md5] = {
            "name": name,
            "date": now(),  # last use time
            "size": size,
            "query": 0,
            "age": self._age,
        }
        self.__save()

    def __save(self):
        with open(self._recorder, "w") as fb:
            fb.write(f"# {now()}\n\n")
            fb.write("record = ")
            pprint.pprint(self.record, width=80, stream=fb)

    def __aging(self):
        # https://en.wikipedia.org/wiki/Page_replacement_algorithm#Aging
        for v in self.record.values():
            v["age"] /= 2

    def is_full(self):
        return self.size > self._max_capacity

    def eviction(self):
        # maintain file recorder
        # remove folder
        print("Maintain files.")
        file_alive = set()
        for f in os.listdir(self._folder):
            _f = os.path.join(self._folder, f)
            if os.path.isdir(_f):
                try:
                    shutil.rmtree(_f)
                except:
                    pass
                continue
            if f not in self.record:
                os.remove(_f)
                continue
            file_alive.add(f)

        invalid_record = self.record.keys() - file_alive
        for k in invalid_record:
            self.size -= self.record[k]["size"]
            del self.record[k]

        if not self.is_full():
            self.__save()
            return

        # remove some files if storage shortage
        for k, _ in sorted(self.record.items(), key=lambda x: x[1]["age"]):
            os.remove(os.path.join(self._folder, k))
            self.size -= self.record[k]["size"]
            del self.record[k]
            if self.size < self._max_capacity:
                break
        self.__save()


files = FileRecorder()


@atexit.register
def save_cache():
    files.eviction()
    print("record saved. Goodbye")


def receve_file(md5, name, buffer):
    if files.is_full():
        files.eviction()
    files[md5] = (name, buffer)
    print(f"received file '{name}'")


def has_file(md5):
    return md5 in files


def xor_md5(md5s):
    xormd5 = int(md5s[0], base=16)
    for m in md5s[1:]:
        xormd5 ^= int(m, base=16)
    return f"{xormd5:x}"


def build_dir(md5):
    # use XOR MD5 as the folder name to keep it stable.
    folder_name = xor_md5(list(md5.values()))
    folder = os.path.join(FileRecorder._folder, folder_name)
    if os.path.exists(folder):
        print(f"folder {folder} exists.")
        return folder_name
    os.makedirs(folder)
    for name, md5 in md5.items():
        os.symlink(f"../{md5}", os.path.join(folder, name))
    return folder_name


def run_command(cmd_fmt: str, *md5):
    if all(md5):
        cmd_fmt = cmd_fmt.format(*(files[x] for x in md5))

    out = subprocess.run(
        cmd_fmt,
        shell=True,
        capture_output=True,
    )
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-p",
        "--port",
        type=int,
        default=8000,
        dest="port",
        help="server port number, default:8000",
    )
    args = parser.parse_args()
    server = SimpleXMLRPCServer(("0.0.0.0", args.port), allow_none=True)
    print(f"Listening on port {args.port}...")
    server.register_function(receve_file, "send_file")
    server.register_function(build_dir, "build_dir")
    server.register_function(run_command, "run")  # type: ignore
    server.register_function(has_file, "has_file")
    server.serve_forever()
