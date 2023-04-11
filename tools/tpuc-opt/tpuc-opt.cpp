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

#include "tpu_mlir/InitAll.h"

#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
using namespace mlir;

const std::string PluginPrePass[] = {"--init"};

const std::string PluginPostPass[] = {"--deinit", "--mlir-print-debuginfo"};

int main(int argc, char **argv) {
  tpu_mlir::registerAllPasses();

  DialectRegistry registry;
  tpu_mlir::registerAllDialects(registry);
  if (argc <= 2) {
    return asMainReturnCode(
        MlirOptMain(argc, argv, "TPU MLIR module optimizer driver\n", registry,
                    /*preloadDialectsInContext=*/false));
  }

  int num_pre = sizeof(PluginPrePass) / sizeof(PluginPrePass[0]);
  int num_post = sizeof(PluginPostPass) / sizeof(PluginPostPass[0]);
  int new_argc = num_pre + argc + num_post;
  char *new_argv[new_argc];
  int left = 0;
  int idx = 0;
  for (; left < argc; left++) {
    if (strncmp(argv[left], "--", 2) == 0) {
      break;
    }
    new_argv[idx] = argv[left];
    idx++;
  }
  new_argv[0] = argv[0];
  new_argv[1] = argv[1];
  for (int i = 0; i < num_pre; i++) {
    new_argv[idx] = (char *)PluginPrePass[i].c_str();
    idx++;
  }
  for (; left < argc; left++) {
    if (std::string(argv[left]) == "-o") {
      break;
    }
    new_argv[idx] = argv[left];
    idx++;
  }
  for (int i = 0; i < num_post; i++) {
    new_argv[idx] = (char *)PluginPostPass[i].c_str();
    idx++;
  }
  for (int i = left; i < argc; i++) {
    new_argv[idx] = argv[i];
    idx++;
  }

  return asMainReturnCode(MlirOptMain(
      new_argc, new_argv, "TPU MLIR module optimizer driver\n", registry,
      /*preloadDialectsInContext=*/false));
}
