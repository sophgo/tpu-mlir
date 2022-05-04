# -*- Python -*-

import os
import platform
import re
import subprocess
import tempfile

import lit.formats
import lit.util

from lit.llvm import llvm_config
from lit.llvm.subst import ToolSubst
from lit.llvm.subst import FindTool

# Configuration file for the 'lit' test runner.

# name: The name of this test suite.
config.name = "Sophgo"

config.test_format = lit.formats.ShTest(not llvm_config.use_lit_shell)

# suffixes: A list of file extensions to treat as test files.
config.suffixes = [".td", ".mlir", ".ll", ".tc", ".py", ".yaml", ".test", ".pdll", ".c"]

# test_source_root: The root path where tests are located.
config.test_source_root = os.path.dirname(__file__)

# test_exec_root: The root path where tests should be run.
config.test_exec_root = os.path.join(config.mlir_obj_root, "test")

config.substitutions.append(("%PATH%", config.environment["PATH"]))
config.substitutions.append(("%shlibext", config.llvm_shlib_ext))
config.substitutions.append(("%mlir_src_root", config.mlir_src_root))
config.substitutions.append(("%host_cxx", config.host_cxx))
config.substitutions.append(("%host_cc", config.host_cc))

llvm_config.with_system_environment(["HOME", "INCLUDE", "LIB", "TMP", "TEMP"])

llvm_config.use_default_substitutions()

# excludes: A list of directories to exclude from the testsuite. The 'Inputs'
# subdirectories contain auxiliary inputs for various tests in their parent
# directories.
config.excludes = [
    "Inputs",
    "CMakeLists.txt",
    "README.txt",
    "LICENSE.txt",
    "lit.cfg.py",
    "lit.site.cfg.py",
]

# test_source_root: The root path where tests are located.
config.test_source_root = os.path.dirname(__file__)

# test_exec_root: The root path where tests should be run.
config.test_exec_root = os.path.join(config.mlir_obj_root, "test")

# Tweak the PATH to include the tools dir.
llvm_config.with_environment("PATH", config.sophgo_tools_dir, append_path=True)
llvm_config.with_environment("PATH", config.llvm_tools_dir, append_path=True)
# llvm_config.with_environment("PATH", config.sophgo_py_tools_dir, append_path=True)

tool_dirs = [config.sophgo_tools_dir, config.llvm_tools_dir]
tools = [
    "sophgo-opt",
    # "model_transform.py",
]


python_executable = config.python_executable
# Python configuration with sanitizer requires some magic preloading. This will only work on clang/linux.
# TODO: detect Darwin/Windows situation (or mark these tests as unsupported on these platforms).
if "asan" in config.available_features and "Linux" in config.host_os:
    python_executable = f"LD_PRELOAD=$({config.host_cxx} -print-file-name=libclang_rt.asan-{config.host_arch}.so) {config.python_executable}"
tools.extend(
    [
        ToolSubst("%PYTHON", python_executable, unresolved="ignore"),
    ]
)

llvm_config.add_tool_substitutions(tools, tool_dirs)


# FileCheck -enable-var-scope is enabled by default in MLIR test
# This option avoids to accidentally reuse variable across -LABEL match,
# it can be explicitly opted-in by prefixing the variable name with $
config.environment["FILECHECK_OPTS"] = "-enable-var-scope --allow-unused-prefixes=false"


if config.native_target in config.targets_to_build:
    config.available_features.add("llvm_has_native_target")


# Add the python path for both the source and binary tree.
# Note that presently, the python sources come from the source tree and the
# binaries come from the build tree. This should be unified to the build tree
# by copying/linking sources to build.
# if config.enable_bindings_python:
#     llvm_config.with_environment(
#         "PYTHONPATH",
#         [
#             os.path.join(config.mlir_obj_root, "python_packages", "mlir_core"),
#             os.path.join(config.mlir_obj_root, "python_packages", "mlir_test"),
#         ],
#         append_path=True,
#     )

if config.enable_assertions:
    config.available_features.add("asserts")
else:
    config.available_features.add("noasserts")
