//===- mlir-opt.cpp - MLIR Optimizer Driver -------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Main entry function for mlir-opt for when built as standalone binary.
//
//===----------------------------------------------------------------------===//

#include "include/Utils.h"
using namespace llvm;
using namespace mlir;

int main(int argc, char **argv) {
  static llvm::cl::OptionCategory MlirOptions("Frontend Options", "");

  llvm::cl::opt<std::string> input_filename(
      llvm::cl::Positional, llvm::cl::desc("<input file>"), cl::init("-"),
      llvm::cl::cat(MlirOptions));

  llvm::cl::opt<std::string> output_filename(
      "o", llvm::cl::desc("Output filename"), llvm::cl::value_desc("filename"),
      llvm::cl::init("-"), llvm::cl::cat(MlirOptions));

  llvm::cl::opt<std::string> chip_name("chip", llvm::cl::desc("Chip name"),
                                       llvm::cl::init("BM1684X"),
                                       llvm::cl::cat(MlirOptions));

  llvm::cl::opt<bool> split_input_file(
      "split-input-file",
      llvm::cl::desc("Split the input file into pieces and process each "
                     "chunk independently"),
      llvm::cl::init(false), llvm::cl::cat(MlirOptions));

  llvm::cl::opt<bool> verify_diagnostics(
      "verify-diagnostics",
      llvm::cl::desc("Check that emitted diagnostics match "
                     "expected-* lines on the corresponding line"),
      llvm::cl::init(false), llvm::cl::cat(MlirOptions));

  llvm::cl::opt<bool> verify_passes(
      "verify-each",
      llvm::cl::desc("Run the verifier after each transformation pass"),
      llvm::cl::init(false), llvm::cl::cat(MlirOptions));

  llvm::cl::opt<bool> allowUnregisteredDialects(
      "allow-unregistered-dialect",
      llvm::cl::desc("Allow operation with no registered dialects"),
      llvm::cl::init(false), llvm::cl::cat(MlirOptions));

  llvm::cl::opt<bool> dynamic_mode(
      "Dynamic mode", llvm::cl::desc("Dynamic mode"), llvm::cl::init(false),
      llvm::cl::cat(MlirOptions));

  llvm::InitLLVM y(argc, argv);
  registerAllPasses();
  DialectRegistry registry;
  registerAllDialects(registry);
  registerAllExtensions(registry);
  // Register MLIR command line options.
  mlir::registerAsmPrinterCLOptions();
  mlir::registerMLIRContextCLOptions();
  mlir::registerPassManagerCLOptions();
  mlir::registerDefaultTimingManagerCLOptions();
  mlir::registerAsmPrinterCLOptions();

  cl::ParseCommandLineOptions(argc, argv, "tpuc-opt-exp compiler\n");
  PassPipelineCLParser passPipeline("", "Compiler passes to run");
  // Set up the input file.
  std::string error_message;
  auto file = openInputFile(input_filename, &error_message);
  if (!error_message.empty()) {
    llvm::errs() << "Failure to open file; " << error_message << "\n";
    return 1;
  }

  auto output = openOutputFile(output_filename, &error_message);
  if (!error_message.empty()) {
    llvm::errs() << "Failure to compile file; " << error_message << "\n";
    return 1;
  }

  auto passManagerSetupFn = [&](PassManager &pm) {
    MLIRContext *ctx = pm.getContext();
    // MlirOptMain constructed ctx with our registry so we just load all our
    // already registered dialects.
    ctx->loadAllAvailableDialects();
    buildPrecompileTransformPassPipeline(pm, chip_name, dynamic_mode);
    return success();
  };

  MlirOptMainConfig config;
  config.setPassPipelineSetupFn(passManagerSetupFn)
      .splitInputFile(split_input_file)
      .verifyDiagnostics(verify_diagnostics)
      .verifyPasses(verify_passes)
      .allowUnregisteredDialects(allowUnregisteredDialects)
      .emitBytecode(false)
      .useExplicitModule(false);

  if (failed(MlirOptMain(output->os(), std::move(file), registry, config)))
    return 1;

  output->keep();
  return 0;
}
