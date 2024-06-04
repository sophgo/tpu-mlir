#!/usr/bin/env python3
# ==============================================================================
#
# Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
#
# TPU-MLIR is licensed under the 2-Clause BSD License except for the
# third-party components.
#
# ==============================================================================

import time

class Timer:

    def __init__(self):
        self.start_time = time.time()

    def elapsed_time(self):
        return time.time() - self.start_time
