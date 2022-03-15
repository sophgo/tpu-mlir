#pragma once

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"

using namespace mlir;

namespace sophgo {

// =======================
// interfece for moduleop
// =======================
// TODO(pengchao.hu): define state by enum in td
mlir::ModuleOp getModuleOp(mlir::Operation*op);
llvm::StringRef getMlirWeightFile(mlir::ModuleOp module);
void setMlirWeightFile(mlir::ModuleOp module, llvm::StringRef weight_file);
llvm::StringRef getMlirState(mlir::ModuleOp module);
void setMlirState(mlir::ModuleOp module, llvm::StringRef state);
llvm::StringRef getMlirChip(mlir::ModuleOp module);
void setMlirChip(mlir::ModuleOp module, llvm::StringRef chip);

}
