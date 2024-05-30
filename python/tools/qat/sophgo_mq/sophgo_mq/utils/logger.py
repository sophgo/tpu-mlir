import logging
import sys


SOPHGO_MQ_LOGGER_NAME = "SOPHGO_MQ"
logger = logging.getLogger(SOPHGO_MQ_LOGGER_NAME)
logger.propagate = False
stdout_handler = logging.StreamHandler(sys.stdout)
fmt = logging.Formatter("[%(name)s] %(levelname)s: %(message)s")
stdout_handler.setFormatter(fmt)
stdout_handler.setLevel(logging.DEBUG)
logger.addHandler(stdout_handler)
logger.setLevel(logging.INFO)
logger.parent = None


def set_log_level(level):
    logger.setLevel(level)
    for handler in logger.handlers:
        handler.setLevel(level)


def disable_logging():
    logger.handlers = []
