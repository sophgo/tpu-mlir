#!/usr/bin/env python3
# Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
#
# TPU-MLIR is licensed under the 2-Clause BSD License except for the
# third-party components.
#
# ==============================================================================

# strip the coeff data and kernel_module, keep machine code only.

from debugger import disassembler


def remove_weight(bmodel):
    bmodel.kernel_module.has = False
    bmodel.net[0].parameter[0].coeff_mem.has = False
    bmodel.net[0].parameter[0].net_profile.has = False
    return


if __name__ == "__main__":
    import sys

    bmodel = disassembler.BModel(sys.argv[1])
    remove_weight(bmodel)
    bmodel.serialize(sys.argv[2])
