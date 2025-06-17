#!/usr/bin/env python3
# Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
#
# TPU-MLIR is licensed under the 2-Clause BSD License except for the
# third-party components.
#
# ==============================================================================
from pprint import pprint
import requests
import os
import json
import time
import json
import traceback

try:
    ret = requests.post(
        "http://172.28.142.50:8502/gerrit",
        data=json.dumps({"env": os.environ.copy()}, ),
    )
    pprint(json.loads(ret.content))
    for i in range(10):
        ret = requests.post(
            "http://172.28.142.50:8502/gerrit/status",
            data=json.dumps({"env": os.environ.copy()}),
        )
        pprint(ret.content)
        time.sleep(6)
except Exception:
    traceback.print_exc()
    pass
