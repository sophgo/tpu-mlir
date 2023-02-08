#!/usr/bin/env python3
# ==============================================================================
#
# Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
#
# TPU-MLIR is licensed under the 2-Clause BSD License except for the
# third-party components.
#
# ==============================================================================

from soc_rpc.client import SOCClient
import sys


if __name__ == "__main__":
    client = SOCClient("http://172.28.9.198:8000/")
    # client = SOCClient("http://172.28.3.138:8000/")
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
