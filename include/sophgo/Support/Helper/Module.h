#pragma once

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"

using namespace mlir;

namespace sophgo {
namespace helper {
struct Module {
  struct Attr {
    static constexpr llvm::StringRef NAME = "module.name";
    static constexpr llvm::StringRef STATE = "module.state";
    static constexpr llvm::StringRef CHIP = "module.chip";
    static constexpr llvm::StringRef WEIGHT_FILE = "module.weight_file";
  };

  struct State {
    static constexpr llvm::StringRef TOP_F32 = "TOP_F32";
    static constexpr llvm::StringRef TOP_CALIBRATED = "TOP_CALIBRATED";
    static constexpr llvm::StringRef TOP_QUANTIZED = "TOP_QUANTIED";
    static constexpr llvm::StringRef TPU_QUANTIZED = "TPU_QUANTIED";
  };

  static ModuleOp getModuleOp(Operation *op);
  static llvm::StringRef getWeightFile(ModuleOp module);
  static void setWeightFile(ModuleOp module, llvm::StringRef weight_file);
  static llvm::StringRef getState(ModuleOp module);
  static void setState(ModuleOp module, llvm::StringRef state);
  static bool isState(ModuleOp module, llvm::StringRef state);
  static llvm::StringRef getChip(ModuleOp module);
  static void setChip(ModuleOp module, llvm::StringRef chip);
};
} // namespace helper
} // namespace sophgo
