#!/usr/bin/env python3
# ==============================================================================
#
# Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
#
# TPU-MLIR is licensed under the 2-Clause BSD License except for the
# third-party components.
#
# ==============================================================================
import os
import sys
import logging


log_name = dict()
def setup_logger(name, log_level="INFO"):
    if name in log_name:
        return log_name[name]

    formatter = logging.Formatter(
        datefmt='%Y/%m/%d %H:%M:%S', fmt='%(asctime)s - %(levelname)s : %(message)s')

    handler = logging.StreamHandler(stream=sys.stderr)
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    if log_level == "DEBUG":
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)
    logger.addHandler(handler)
    log_name[name] = logger
    return logger

class logger():
    def __init__(self, log_file_name:str, log_level:str = "DEBUG"):
        root_logger = logging.getLogger()
        for h in root_logger.handlers:
            root_logger.removeHandler(h)
        os.system(f'rm -f {log_file_name}')
        level = logging.DEBUG
        if log_level == "INFO":
            level = logging.INFO
        logging.basicConfig(filename=log_file_name, level=level)

    def print_dbg(self, *para):
        tmp = [str(item) for item in para]
        tmpStr = ' '.join(tmp)
        print(tmpStr)
        logging.debug(tmpStr)

    def print_info(self, *para):
        tmp = [str(item) for item in para]
        tmpStr = ' '.join(tmp)
        print(tmpStr)
        logging.info(tmpStr)

