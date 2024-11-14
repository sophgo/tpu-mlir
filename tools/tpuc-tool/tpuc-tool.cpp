//===- mlir-opt.cpp - MLIR Optimizer Driver -------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//
//
// Main entry function for mlir-opt for when built as standalone binary.
//
//===----------------------------------------------------------------------===//

#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "tpu_mlir/InitAll.h"
using namespace mlir;

std::vector<std::string> pluginPostPass = {
    "--mlir-print-debuginfo", // keep locations in mlir
};

int main(int argc, char **argv) {
  tpu_mlir::registerToolPasses();

  DialectRegistry registry;
  tpu_mlir::registerAllDialects(registry);

  const int num_post = pluginPostPass.size();
  const int new_argc = argc + num_post;
  char *new_argv[new_argc];

  int idx = 0;
  int left = 0;
  for (; left < argc; left++) {
    if (strncmp(argv[left], "--", 2) == 0) {
      break;
    }
    new_argv[idx] = argv[left];
    idx++;
  }
  for (int i = 0; i < num_post; i++) {
    new_argv[idx] = (char *)pluginPostPass[i].c_str();
    idx++;
  }
  for (int i = left; i < argc; i++) {
    new_argv[idx] = argv[i];
    idx++;
  }

  return asMainReturnCode(MlirOptMain(
      new_argc, new_argv, "TPU MLIR module optimizer driver\n", registry));
}
