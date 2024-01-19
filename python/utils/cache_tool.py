import json
import os
from typing import Optional
from .log_setting import setup_logger
from tools.bmodel_dis import dis, BModel2MLIR
import pickle
import hashlib
import datetime

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
                "open cache_skip mode, validation will skipped when hash comparison succeed."
            )

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

        if (
            self["top_mlir_debug"] == debug
            and self["top_mlir"] == new_hash
            and not self.sim_more_strict(self["top_sim"], sim)
            and self["top_validate"]
            and os.path.exists(top_output)
            and self.cache_skip
        ):
            self["top_sim"] = sim  # always update last used similarity
            logger.info("cached skip top validatation.")
            return False
        self["top_sim"] = sim
        self["top_mlir_debug"] = debug
        self["top_mlir"] = new_hash
        return True

    def do_tpu_validate(self, tpu_mlir, tpu_output, sim, debug=False):
        new_hash = self.hash_file(tpu_mlir)
        if (
            self["tpu_mlir_debug"] == debug
            and self["tpu_mlir"] == new_hash
            and not self.sim_more_strict(self["tpu_sim"], sim)
            and self["tpu_validate"]
            # and os.path.exists(tpu_output)
            and self.cache_skip
        ):
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

        if (
            self["model_mlir"] == new_hash
            and self["model_validate"]
            # and os.path.exists(model_output)
            and self.cache_skip
        ):
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
