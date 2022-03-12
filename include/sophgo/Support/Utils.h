#pragma once

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LLVM.h"

namespace mlir {

// =======================
// interfece for moduleop
// =======================
// TODO(pengchao.hu): define state by enum in td
llvm::StringRef getMlirWeightFile(mlir::ModuleOp module);
llvm::StringRef getMlirState(mlir::ModuleOp module);
void setMlirState(mlir::ModuleOp module, llvm::StringRef state);
llvm::StringRef getMlirChip(mlir::ModuleOp module);
void setMlirChip(mlir::ModuleOp module, llvm::StringRef chip);

// =======================
// interfece for quantization
// =======================
void get_scale_and_shift(float scale_f, int& scale, int& shift,int bitwidth=32);
void get_scale_and_shift_positive(float scale_f, int& scale, int& shift,int bitwidth=32);
void get_scale_and_shift_positive_maxshift(float scale_f, int& scale, int& shift,int bitwidth, int max_shift=8);

}

