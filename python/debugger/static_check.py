# ==============================================================================
#
# Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
#
# TPU-MLIR is licensed under the 2-Clause BSD License except for the
# third-party components.
#
# ==============================================================================
from typing import Dict, Type
from .tdb_support import TdbCmdBackend


class Check:
    name: str

    def __init_subclass__(cls) -> None:
        checks[cls.name] = cls


checks: Dict[str, Type[Check]] = {}


class Checker:
    def __init__(self, tdb: TdbCmdBackend) -> None:
        self.tdb = tdb

    def do_checker(self, *names: str):
        for name in names:
            checks[name]().check(self.tdb)

    def check_list(self):
        return list(checks)

    def __repr__(self) -> str:
        return f"supported check: {self.check_list()}"
