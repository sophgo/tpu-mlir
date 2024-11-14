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
#include <fstream>

#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "tpu_mlir/InitAll.h"
using namespace mlir;

const std::string PluginPrePass[] = {"--init"};
std::vector<std::string> PluginPostPass;

int main(int argc, char **argv) {
  tpu_mlir::registerAllPasses();

  DialectRegistry registry;
  tpu_mlir::registerAllDialects(registry);

  // Check if --init option is explicitly provided
  bool hasDeinit = false;
  for (int i = 1; i < argc; ++i) {
    if (std::string(argv[i]).find("--deinit") != std::string(argv[i]).npos) {
      hasDeinit = true;
      break;
    }
  }

  if (hasDeinit) {
    PluginPostPass = {"--mlir-print-debuginfo"};
  } else {
    PluginPostPass = {"--deinit", "--mlir-print-debuginfo"};
  }

  if (argc <= 2) {
    return asMainReturnCode(MlirOptMain(
        argc, argv, "TPU MLIR module optimizer driver\n", registry));
  }

  std::string debug_cmd = argv[argc - 1];
  std::string substring = "--debug_cmd=";
  if (debug_cmd.find(substring) != std::string::npos) {
    std::ofstream ofs;
    ofs.open("/tmp/debug_cmd", std::ios::out | std::ios::trunc);
    ofs << debug_cmd.substr(substring.size()) << std::endl;
    argc -= 1;
  }

  int num_pre = sizeof(PluginPrePass) / sizeof(PluginPrePass[0]);
  int num_post = PluginPostPass.size();
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
    if (strncmp(argv[left], PluginPrePass[i].c_str(),
                PluginPrePass[i].length()) == 0) {
      new_argc--;
      continue;
    }
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
      new_argc, new_argv, "TPU MLIR module optimizer driver\n", registry));
}
