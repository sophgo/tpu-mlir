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


def wrap_print(logger: logging.Logger):
    def log_print(*args, sep=" "):
        logger.info(sep.join([f"{arg}" for arg in args]))

    return log_print


def setup_logger(name, log_level="INFO"):
    if name in log_name:
        return log_name[name]

    logger = logging.getLogger(name)
    logger.setLevel(logging._nameToLevel[log_level])

    formatter = logging.Formatter(
        datefmt="%Y/%m/%d %H:%M:%S", fmt="%(asctime)s - %(levelname)s : %(message)s"
    )


    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(logging.INFO)
    stdout_handler.setFormatter(formatter)
    logger.addHandler(stdout_handler)

    log_name[name] = logger
    logger.parent = None
    # logger.print = wrap_print(logger)
    return logger


class logger:
    def __init__(self, log_file_name: str, log_level: str = "DEBUG"):
        root_logger = logging.getLogger()
        for h in root_logger.handlers:
            root_logger.removeHandler(h)
        os.system(f"rm -f {log_file_name}")
        level = logging.DEBUG
        if log_level == "INFO":
            level = logging.INFO
        logging.basicConfig(filename=log_file_name, level=level)

    def print_dbg(self, *para):
        tmp = [str(item) for item in para]
        tmpStr = " ".join(tmp)
        print(tmpStr)
        logging.debug(tmpStr)

    def print_info(self, *para):
        tmp = [str(item) for item in para]
        tmpStr = " ".join(tmp)
        print(tmpStr)
        logging.info(tmpStr)
