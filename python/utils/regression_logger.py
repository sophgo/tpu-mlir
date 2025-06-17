from contextlib import redirect_stdout, redirect_stderr
import functools
import io
import logging
import os
import sys
import traceback
import pdb
import functools
import traceback
from contextlib import contextmanager

regression_out_path = os.path.join(os.getenv("REGRESSION_PATH"), "regression_out")
assert regression_out_path
regression_log_path = os.path.join(regression_out_path, "../regression_op_log")
os.makedirs(regression_log_path, exist_ok=True)

try:
    import ctypes
    from ctypes.util import find_library
except ImportError:
    libc = None
else:
    try:
        libc = ctypes.cdll.LoadLibrary(find_library("c"))
    except Exception:
        libc = None


class TestFailError(Exception):
    pass


@contextmanager
def stdout_stderr_redirected(to=os.devnull, stdout=None, stderr=None):
    if stdout is None:
        stdout = sys.stdout
    if stderr is None:
        stderr = sys.stderr

    stdout_fd = stdout.fileno()
    stderr_fd = stderr.fileno()
    with os.fdopen(os.dup(stdout_fd), "wb") as copied_stdout, os.fdopen(os.dup(stderr_fd),
                                                                        "wb") as copied_stderr:
        stdout.flush()
        stderr.flush()
        if libc:
            libc.fflush(None)

        try:
            os.dup2(to.fileno(), stdout_fd)
            os.dup2(to.fileno(), stderr_fd)
        except ValueError:
            with open(to, "wb") as to_file:
                os.dup2(to_file.fileno(), stdout_fd)
                os.dup2(to_file.fileno(), stderr_fd)
        try:
            yield stdout, stderr
        finally:
            stdout.flush()
            stderr.flush()
            if libc:
                libc.fflush(None)
            os.dup2(copied_stdout.fileno(), stdout_fd)
            os.dup2(copied_stderr.fileno(), stderr_fd)


def run_in_log_wrapper(func):

    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        concise_log = getattr(self, "concise_log", False)
        if not concise_log:
            return func(self, *args, **kwargs)

        current_path = os.path.abspath(args[0])
        log_path = os.path.join(regression_log_path,
                                os.path.relpath(current_path, regression_out_path))
        os.makedirs(log_path, exist_ok=True)
        running_log = os.path.join(log_path, "regression.log")

        print("Test: {}".format(args[0]))
        with open(running_log, "w", buffering=1) as f:
            with stdout_stderr_redirected(to=f, stdout=sys.stdout, stderr=sys.stderr):
                try:
                    return func(self, *args, **kwargs)
                except Exception as e:
                    console_err_trace = traceback.format_exc()
                    f.write(console_err_trace)
            print(f"------------ Error occurs in: {args[0]}, log in: {running_log} ------------")
            print(console_err_trace)
            raise TestFailError

    return wrapper
