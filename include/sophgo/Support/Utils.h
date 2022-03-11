#pragma once

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LLVM.h"

namespace mlir {

// TODO(pengchao.hu): define state by enum in td

llvm::StringRef getMlirWeightFile(mlir::ModuleOp module);
llvm::StringRef getMlirState(mlir::ModuleOp module);
void setMlirState(mlir::ModuleOp module, llvm::StringRef state);
llvm::StringRef getMlirChip(mlir::ModuleOp module);
void setMlirChip(mlir::ModuleOp module, llvm::StringRef chip);
}
