# Copyright (C) 2025 Sophgo Technologies Inc.  All rights reserved.
#
# TPU-MLIR is licensed under the 2-Clause BSD License except for the
# third-party components.
#
# ==============================================================================
from safetensors import safe_open
import os, torch


class LlmLoad:

    def __init__(self, model_path: str):
        self.st_files = []
        # get all safetensors
        for entry in os.listdir(model_path):
            file_path = os.path.join(model_path, entry)
            if os.path.isfile(file_path):
                if entry.lower().endswith('.safetensors'):
                    f = safe_open(file_path, "pt")
                    self.st_files.append(f)
                elif entry.lower().endswith('.bin'):
                    f = torch.load(file_path, map_location="cpu")
                    self.st_files.append(f)
        # build key -> (file_handle, is_dict) index for O(1) lookup
        self.key_to_file = {}
        for f in self.st_files:
            is_dict = isinstance(f, dict)
            for key in f.keys():
                self.key_to_file[key] = (f, is_dict)

    def read(self, key: str):
        if key not in self.key_to_file:
            raise RuntimeError(f"Can't find key: {key}")
        f, is_dict = self.key_to_file[key]
        if is_dict:
            data = f[key]
        else:
            data = f.get_tensor(key)
        if data.dtype in [torch.float16, torch.bfloat16, torch.float8_e4m3fn, torch.float8_e5m2]:
            return data.float().numpy()
        return data.numpy()

    def is_exist(self, key: str):
        return key in self.key_to_file
