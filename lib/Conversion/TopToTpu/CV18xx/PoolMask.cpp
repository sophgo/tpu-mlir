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

namespace tpu_mlir {
namespace cv18xx {

void PoolMaskLowering::LoweringINT8(PatternRewriter &rewriter,
                                     top::PoolMaskOp op,
                                     bool asymmetric) const {
  std::vector<NamedAttribute> attrs;
  for (auto &attr : op->getAttrs()) {
    attrs.emplace_back(attr);
  }
  auto mask_shape = Module::getShape(op.output());
  auto type = op.output().getType().cast<RankedTensorType>();
  auto quant_type =
      quant::CalibratedQuantizedType::get(type.getElementType(), -127, 127);
  auto new_type = RankedTensorType::get(type.getShape(), quant_type);
  auto pool_mask_type = RankedTensorType::get(mask_shape, quant_type);

  rewriter.replaceOpWithNewOp<tpu::PoolMaskOp>(op, pool_mask_type,
                                               ValueRange{op.input()}, attrs);
}

void PoolMaskLowering::LoweringBF16(PatternRewriter &rewriter,
                                     top::PoolMaskOp op) const {
  lowering_common_bf16<tpu::PoolMaskOp>(rewriter, op);
}

} // namespace cv18xx
} // namespace tpu_mlir
