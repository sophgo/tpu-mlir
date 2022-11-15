//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//
#include "tpu_mlir/Conversion/TopToTpu/LoweringCV18xx.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "lowering-relu"
namespace tpu_mlir {
namespace cv18xx {

void ReluLowering::LoweringINT8(PatternRewriter &rewriter, top::ReluOp op,
                                bool asymmetric) const {
  assert(!asymmetric && "CV18xx not support asymmetric quantify");
  lowering_common_int8<tpu::ReluOp>(rewriter, op, asymmetric);
}

void ReluLowering::LoweringBF16(PatternRewriter &rewriter,
                                top::ReluOp op) const {
  lowering_common_bf16<tpu::ReluOp>(rewriter, op);
}
}
}
