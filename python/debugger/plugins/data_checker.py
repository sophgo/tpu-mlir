# ==============================================================================
#
# Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
#
# TPU-MLIR is licensed under the 2-Clause BSD License except for the
# third-party components.
#
# ==============================================================================

import numpy as np
import cmd
from typing import Tuple
from ..target_common import MemRefBase
from ..final_mlir import Value
from numpy_helper.npz_compare import TensorCompare
from ..tdb_support import TdbCmdBackend, TdbPlugin
from ..final_mlir import Value as MlirValue
from dataclasses import dataclass


@dataclass()
class ValueResult:
    cmp: Tuple
    memref: MemRefBase
    loc_opd: Value
    actual: np.ndarray
    desired: np.ndarray
    zero_point: 0
    scale: 1


class DataCheck(TdbPlugin, cmd.Cmd):
    """
    DataCheck
    """

    name = "data-check"
    func_names = ["compare"]

    def __init__(self, tdb: TdbCmdBackend) -> None:
        super().__init__(tdb)

        self.ref_data = []
        for ref_fn in tdb.reference_data_fns:
            self.ref_data.append(np.load(ref_fn))

        # import pdb; pdb.set_trace()
        self.tc = TensorCompare(
            cosine_similarity_tol=0.99, euclidean_similarity_tol=0.9
        )

        self.failed_operands = {}
        self.failed_results = {}

    def do_report(self, arg):
        pass

    def get_ref_data(self, operand: MlirValue):
        for ref in self.ref_data:
            if operand.name not in ref:
                continue
            reshape = operand.reshape
            ref_data = ref[operand.name]

            if reshape:
                reshape = eval(reshape[1:-1].replace("x", ","))  # '1,3,1,192,1024'
                ref_data = ref_data.reshape(reshape)

            _slice = operand.slice
            data = eval(f"ref_data{_slice}")  # type: np.ndarray
            # The data in HW has a transposed collapsed shape.
            # To align the Bmodel with TPU.mlir, we need to transpose the reference data.
            if operand.layout in (
                "continuous_group3d",
                "eu_align_group3d",
                "compact_group3d",
                "eu_align_xn_group3d",
                "compact_xn_group3d",
            ):
                n, c, d, h, w = 0, 1, 2, 3, 4
                data = data.transpose((d, n, c, h, w))
            return data

        return None

    def compare(self, tdb: TdbCmdBackend, is_operand: bool):
        op = tdb.get_op()
        index = tdb.cmd2index[op.tuple_key]
        if is_operand:
            index -= 1

        index2loc = tdb.before_index2loc if is_operand else tdb.after_index2loc

        failed_value = self.failed_operands if is_operand else self.failed_results

        if index not in index2loc:
            return

        for loc_index in index2loc[index]:
            loc = tdb.final_mlir.loc[loc_index]

            loc_values = loc.operands if is_operand else loc.results
            op_values = op.operands if is_operand else op.results
            opds = {
                opd_loc.address: (i, opd_loc)
                for i, opd_loc in enumerate(loc_values)
                if opd_loc is not None
            }
            visited = {}
            for opd_memref in op_values:
                if opd_memref.is_scalar or opd_memref.r_addr not in opds:
                    continue
                opd_id, loc_opd = opds[opd_memref.r_addr]
                if opd_memref.name in visited:
                    # skip compare dumplicated opd, eg. the second a in `b = mul(a,a)`
                    continue
                visited[opd_memref.name] = opd_id

                try:
                    raw_data = tdb.context.memory.get_data(
                        opd_memref
                    )  # type: np.ndarray
                    # TODO change dtype by loc dtype
                    raw_data = raw_data.view(loc_opd.dtype)
                except ValueError:
                    continue

                actual = (
                    raw_data.astype(np.float32) - loc_opd.zero_point
                ) * loc_opd.scale

                desired = self.get_ref_data(loc_opd)
                if desired is None:
                    continue
                actual = actual.reshape(desired.shape)
                cmp_res = self.tc.compare(actual, desired, 1)

                if not cmp_res[0]:
                    value_res = ValueResult(
                        cmp_res,
                        opd_memref,
                        loc_opd,
                        actual,
                        desired,
                        zero_point=loc_opd.zero_point,
                        scale=loc_opd.scale,
                    )
                    failed_value[index] = value_res

    def before_next(self, tdb: TdbCmdBackend):
        if self.ref_data is None:
            return
        self.compare(tdb, True)

    def after_next(self, tdb: TdbCmdBackend):
        if self.ref_data is None:
            return
        self.compare(tdb, False)
