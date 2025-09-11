# ==============================================================================
#
# Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
#
# TPU-MLIR is licensed under the 2-Clause BSD License except for the
# third-party components.
#
# ==============================================================================
import pandas as pd

from include.summary import Summary


def generate_summary(custom_layers, writer, chip_arch):
    smy = Summary(writer)
    smy.load(custom_layers, chip_arch)
    smy.write(chip_arch)
