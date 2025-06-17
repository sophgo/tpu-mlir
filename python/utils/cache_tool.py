#!/usr/bin/env python3
# ==============================================================================
#
# Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
#
# TPU-MLIR is licensed under the 2-Clause BSD License except for the
# third-party components.
#
# ==============================================================================

import json
import os
from typing import Optional
from utils.log_setting import setup_logger
from tools.bmodel_dis import dis, BModel2MLIR
import pickle
import hashlib
import datetime
import sys

logger = setup_logger("cache")


def BModel2Hash(bmodel_file):
    try:
        bmodel = dis.BModel(bmodel_file)
        obj = [
            bmodel.chip,
            bmodel.version,
            bmodel.type,
            bmodel.neuron_size,
            bmodel.device_num,
            bmodel.core_num,
        ]

        obj.append(bytes(bmodel.binary))
        obj.append(str(BModel2MLIR(bmodel_file)))

        md = hashlib.md5()
        md.update(pickle.dumps(obj))
        return md.hexdigest()
    except Exception:
        return hashlib.md5(str(datetime.datetime.now()).encode()).hexdigest()


class CacheTool:

    def __init__(self, cache_skip: bool) -> None:
        self.fn = os.path.join(os.getcwd(), "build_flag.json")
        self.cache = self.read_json(self.fn)
        self.cache_skip = cache_skip or os.environ.get("CACHE_SKIP", "False") == "True"
        if cache_skip:
            logger.info(
                "open cache_skip mode, validation will skipped when hash comparison succeed.")

    def mark_top_success(self):
        self["top_validate"] = True

    def mark_tpu_success(self):
        self["tpu_validate"] = True

    def mark_model_success(self):
        self["model_validate"] = True

    def sim_more_strict(self, old_sim, new_sim):
        """new_sim bigger than old_sim"""
        if old_sim is None:
            return True
        if new_sim[0] > old_sim[0] and new_sim[1] > old_sim[1]:
            return True
        return False

    def do_top_validate(self, top_mlir, top_output, sim, debug=False):
        new_hash = self.hash_file(top_mlir)

        if (self["top_mlir_debug"] == debug and self["top_mlir"] == new_hash
                and not self.sim_more_strict(self["top_sim"], sim) and self["top_validate"]
                and os.path.exists(top_output) and self.cache_skip):
            self["top_sim"] = sim  # always update last used similarity
            logger.info("cached skip top validatation.")
            return False
        self["top_sim"] = sim
        self["top_mlir_debug"] = debug
        self["top_mlir"] = new_hash
        return True

    def do_tpu_validate(self, tpu_mlir, tpu_output, sim, debug=False):
        new_hash = self.hash_file(tpu_mlir)
        if (self["tpu_mlir_debug"] == debug and self["tpu_mlir"] == new_hash
                and not self.sim_more_strict(self["tpu_sim"], sim) and self["tpu_validate"]
                # and os.path.exists(tpu_output)
                and self.cache_skip):
            self["tpu_sim"] = sim  # always update last used similarity
            logger.info("cached skip tpu validatation.")
            return False
        self["tpu_sim"] = sim
        self["tpu_mlir_debug"] = debug
        self["tpu_mlir"] = new_hash
        return True

    def do_model_validate(self, bmodel, model_output):
        from tools.bmodel_dis import BModel2Reg

        new_hash = BModel2Hash(bmodel)

        if (self["model_mlir"] == new_hash and self["model_validate"]
                # and os.path.exists(model_output)
                and self.cache_skip):
            logger.info("cached skip model validatation.")
            return False

        self["model_mlir"] = new_hash
        return True

    def hash_file(self, file: str) -> Optional[str]:
        if os.path.exists(file) and os.path.isfile(file):
            with open(file, "rb") as r:
                st = r.read()
                return hashlib.md5(st).hexdigest()
        return None

    def read_json(self, fn):
        try:
            with open(fn) as r:
                return json.load(r)
        except Exception:
            return {}

    def replace_key_json(self, fn, **kwargs):
        dic = self.read_json(fn)
        dic.update(kwargs)
        with open(fn, "w") as w:
            json.dump(dic, w)

    def __getitem__(self, k):
        return self.cache.get(k, None)

    def __setitem__(self, k, v):
        self.cache[k] = v
        self.replace_key_json(self.fn, **{k: v})


import os
import time
from pathlib import Path

try:
    import pymlir

    version = pymlir.__version__
except ImportError:
    version = "unknown"

history_commands = []


class CommandRecorder:

    def __init__(self, cache_fn, read=False):
        if read and not os.path.exists(cache_fn):
            raise FileNotFoundError(f"cache file {cache_fn} not found")
        self._dic = {"version": version}
        self.cache_fn = cache_fn
        self.base_dir = os.path.abspath(os.path.dirname(cache_fn))
        if os.path.exists(cache_fn):
            self._dic.update(json.loads(Path(cache_fn).read_text()))

        self.cache_rev = {}
        try:
            for type, val in self._dic.get("files", {}).items():
                self.cache_rev[val["path"]] = val["last_modify"]
        except KeyError:
            logger.error(f"cache file {cache_fn} is invalid")
            os.remove(cache_fn)

    def clear(self):
        self._dic = {"version": version}

    @property
    def dic(self):
        dic = self._dic.copy()
        files = {}
        for file_name, file_info in dic.get("files", {}).items():
            file = os.path.realpath(os.path.join(self.base_dir, file_info["path"]))
            if os.path.exists(file):
                files[file_name] = {
                    "path": file,
                    "last_modify": file_info["last_modify"],
                }
        dic["files"] = files
        return dic

    def add_property(self, **kwargs):
        for key, value in kwargs.items():
            ret = self._dic.setdefault("properties", {})
            ret[key] = value

    def add_file(self, **kwargs):
        for name, file_path in kwargs.items():
            if not isinstance(file_path, str):
                continue
            if os.path.exists(file_path):
                ret = self._dic.setdefault("files", {})
                rel_path = os.path.relpath(file_path, self.base_dir)
                if os.path.isdir(file_path):
                    modify_file = Path(file_path).joinpath(".modify")
                    if not modify_file.exists():
                        modify_file.write_text("1")
                    ret[name] = {
                        "path": rel_path,
                        "last_modify": os.path.getmtime(modify_file),
                    }

                else:
                    ret[name] = {
                        "path": rel_path,
                        "last_modify": os.path.getmtime(file_path),
                    }

    def refresh(self):
        self.update_file(self.cache_fn)

    def add_command(self, **kwargs):
        for stage, command in kwargs.items():
            self._dic.setdefault("commands", {})[stage] = command

    def update_file(self, file_path):
        try:
            load_from = CommandRecorder(file_path)

            ref_dir = os.path.dirname(file_path)
            for file_name, file_info in load_from._dic.get("files", {}).items():
                self.add_file(**{file_name: os.path.join(ref_dir, file_info["path"])})

            for k, v in load_from._dic.items():
                if k == "files":
                    continue
                if isinstance(v, dict) and k in self._dic:
                    self._dic[k].update(v)
                elif isinstance(v, dict):
                    self._dic[k] = v.copy()
                else:
                    self._dic[k] = v

        except Exception:
            logger.error(f"cache file {file_path} is invalid")

    def all_file_no_change(self, *file_names):
        for file_name in file_names:
            if file_name not in self.cache_rev:
                return False
            if self.cache_rev[file_name] != os.path.getmtime(file_name):
                return False
        return True

    def dump(self, fn=None):
        if fn is None:
            fn = self.cache_fn
        with open(fn, "w") as w:
            json.dump(self._dic, w, indent=2, ensure_ascii=False)

    def check(self):
        return self.all_file_no_change(*self.cache_rev.keys())


if __name__ == "__main__":
    fr = CommandRecorder(sys.argv[1] if len(sys.argv) > 1 else "ref_files.json")
    print(fr.check())
