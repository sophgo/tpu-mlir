//===----------------------------------------------------------------------===//
//
// Copyright (c) 2020-2030 by Sophgo Technologies Inc. All rights reserved.
//
// Licensed under the Apache License v2.0.
// See http://www.apache.org/licenses/LICENSE-2.0 for license information.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Conversion/TopToTpu/LoweringCV18xx.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "lowering-gather"

namespace tpu_mlir {
namespace cv18xx {
void GatherLowering::LoweringINT8(PatternRewriter &rewriter, top::GatherOp op,
                                  bool asymmetric) const {
  LoweringBF16(rewriter, op);
}

void GatherLowering::LoweringBF16(PatternRewriter &rewriter,
                                  top::GatherOp op) const {
  std::vector<Value> operands;
  auto inputOp = cast<top::WeightOp>(op.getInput().getDefiningOp());
  operands.push_back(op.getIndices());
  operands.push_back(inputOp.clone_bf16(op));
  std::vector<NamedAttribute> attrs;
  attrs.push_back(rewriter.getNamedAttr("cpu_op_name", rewriter.getStringAttr("embedding")));
  attrs.push_back(rewriter.getNamedAttr("param", rewriter.getDictionaryAttr({})));
  auto newType = getQuantBF16Type(op.getOutput());
  rewriter.replaceOpWithNewOp<tpu::GenericCpuOp>(op.getOperation(), newType,
                                             operands, attrs);
}
}
}
