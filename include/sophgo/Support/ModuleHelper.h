#pragma once

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"

using namespace mlir;

namespace sophgo {

static constexpr llvm::StringRef kModelNameAttrName = "module.name";
static constexpr llvm::StringRef kStateAttrName = "module.state";
static constexpr llvm::StringRef kChipAttrName = "module.chip";
static constexpr llvm::StringRef kWeightFileAttrName = "module.weight_file";

static constexpr llvm::StringRef STATE_TOP_F32 = "TOP_F32";
static constexpr llvm::StringRef STATE_TOP_CALIBRATED = "TOP_CALIBRATED";
static constexpr llvm::StringRef STATE_TOP_QUANTIZED = "TOP_QUANTIED";

// =======================
// interfece for moduleop
// =======================
// TODO(pengchao.hu): define state by enum in td
mlir::ModuleOp getModuleOp(mlir::Operation *op);
llvm::StringRef getMlirWeightFile(mlir::ModuleOp module);
void setMlirWeightFile(mlir::ModuleOp module, llvm::StringRef weight_file);
llvm::StringRef getMlirState(mlir::ModuleOp module);
void setMlirState(mlir::ModuleOp module, llvm::StringRef state);
llvm::StringRef getMlirChip(mlir::ModuleOp module);
void setMlirChip(mlir::ModuleOp module, llvm::StringRef chip);

} // namespace sophgo
