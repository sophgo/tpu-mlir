# -*- Python -*-

import os
import platform
import re
import shutil
import subprocess
import tempfile

import lit.formats
import lit.util

from lit.llvm import llvm_config
from lit.llvm.subst import ToolSubst
from lit.llvm.subst import FindTool

# Configuration file for the 'lit' test runner.

# name: The name of this test suite.
config.name = "TPU-MLIR"

config.test_format = lit.formats.ShTest(not llvm_config.use_lit_shell)

# suffixes: A list of file extensions to treat as test files.
config.suffixes = [".mlir", ".test"]

# test_source_root: The root path where tests are located.
config.test_source_root = os.path.dirname(__file__)

# test_exec_root: The root path where tests should be run.
config.test_exec_root = os.path.join(config.tpumlir_obj_root, "test")

config.substitutions.append(("%PATH%", config.environment["PATH"]))
# config.substitutions.append(("%shlibdir", config.llvm_shlib_dir))

llvm_config.with_system_environment(["HOME", "INCLUDE", "LIB", "TMP", "TEMP"])

llvm_config.use_default_substitutions()

# excludes: A list of directories to exclude from the testsuite. The 'Inputs'
# subdirectories contain auxiliary inputs for various tests in their parent
# directories.
config.excludes = ["CMakeLists.txt", "README.txt", "LICENSE.txt"]

# test_source_root: The root path where tests are located.
config.test_source_root = os.path.dirname(__file__)

# test_exec_root: The root path where tests should be run.
config.test_exec_root = os.path.join(config.tpumlir_obj_root, "test")
config.tpumlir_tools_dir = os.path.join(config.tpumlir_obj_root, "bin")
config.tpumlir_python_tools_dir = os.path.join(config.tpumlir_python_root, "tools")
config.tpumlir_libs_dir = os.path.join(config.tpumlir_obj_root, "lib")

# Tweak the PATH to include the tools dir.
llvm_config.with_environment("PATH", config.llvm_tools_dir, append_path=True)
llvm_config.with_environment("PATH", config.tpumlir_python_tools_dir, append_path=True)

tool_dirs = [
    config.tpumlir_tools_dir,
    config.llvm_tools_dir,
]

tools = [
    "tpuc-opt",
    "tpuc-test",
]


llvm_config.add_tool_substitutions(tools, tool_dirs)

llvm_config.with_environment(
    "PYTHONPATH",
    [
        os.path.join(config.mlir_python_dir, "python_packages"),
        os.path.join(config.mlir_python_dir, "python_packages/mlir_core"),
        os.path.join(config.tpumlir_python_root, "utils"),
        os.path.join(config.tpumlir_python_root, "test"),
        os.path.join(config.tpumlir_install_root, "python"),
    ],
    append_path=True,
)
