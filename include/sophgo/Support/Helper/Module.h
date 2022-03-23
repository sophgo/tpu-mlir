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
    static constexpr llvm::StringRef TPU_WEIGHT_REORDERD = "TPU_WEIGHT_REORDERD";
  };

  struct Chip {
    static constexpr llvm::StringRef ALL = "ALL";
    static constexpr llvm::StringRef BM1684 = "BM1684";
    static constexpr llvm::StringRef BM1686 = "BM1686";
  };

  static top::NoneOp getNoneOp(Operation *op);
  static ModuleOp getModuleOp(Operation *op);
  static void updateModuleTypes(ModuleOp module);
  static std::string genWeightFileName(ModuleOp module);
  static int64_t getAddress(Value v);
  static void getNCHW(Value v, int64_t&n, int64_t&c, int64_t&h, int64_t&w, bool left_align = true);

  static inline llvm::StringRef getName(ModuleOp module) {
    return module->getAttrOfType<StringAttr>(Attr::NAME);
  }
  static inline llvm::StringRef getChip(ModuleOp module) {
    return module->getAttrOfType<StringAttr>(Attr::CHIP);
  }
  static inline void setChip(ModuleOp module, StringRef chip) {
    module->setAttr(Attr::CHIP, StringAttr::get(module.getContext(), chip.lower()));
  }
  static inline StringRef getWeightFile(ModuleOp module) {
    return module->getAttrOfType<StringAttr>(Attr::WEIGHT_FILE);
  }
  static inline void setWeightFile(ModuleOp module, StringRef weight_file) {
    module->setAttr(Attr::WEIGHT_FILE,
                    StringAttr::get(module.getContext(), weight_file));
  }
  static inline StringRef getState(ModuleOp module) {
    return module->getAttrOfType<StringAttr>(Attr::STATE);
  }
  static inline void setState(ModuleOp module, StringRef state) {
    module->setAttr(Attr::STATE, StringAttr::get(module.getContext(), state));
  }
  static inline bool isState(ModuleOp module, llvm::StringRef state) {
    return state == getState(module);
  }
};
} // namespace helper
} // namespace sophgo
