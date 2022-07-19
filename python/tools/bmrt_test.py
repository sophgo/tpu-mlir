#!/usr/bin/env python3
# ==============================================================================
#
# Copyright (c) 2020-2030 by Sophgo Technologies Inc. All rights reserved.
#
# Licensed under the Apache License v2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for license information.
# SPDX-License-Identifier: Apache-2.0
#
# ==============================================================================

from soc_rpc.client import SOCClient
import sys


if __name__ == "__main__":
    client = SOCClient("http://172.28.1.80:8000/")
    args = sys.argv[1:]
    file = None
    if "--bmodel" in args:
        index = args.index("--bmodel")
        file = args[index + 1]
        args[index + 1] = "{}"
    elif "--context_dir" in args:
        index = args.index("--context_dir")
        file = args[index + 1]
        args[index + 1] = "{}"

    out = client.run_file("bmrt_test " + " ".join(args), file)
    sys.stdout.write(out["stdout"].data.decode())
    sys.stderr.write(out["stderr"].data.decode())
    sys.exit(out["returncode"])
