//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Conversion/TopToTpu/LoweringCV18xx.h"

namespace tpu_mlir {
namespace cv18xx {
static mlir::Type createQuantInt8Type(mlir::Type &t) {
  // TODO refine
  auto type = t.cast<RankedTensorType>();
  auto ctx = t.getContext();
  auto cali_type = module::getCalibratedType(t);
  double scale = 1.0;
  int64_t zeropoint = 0;
  int64_t qmin = -128, qmax = 127;
  uint32_t flag = quant::QuantizationFlags::Signed;
  auto qtype = quant::UniformQuantizedType::get(flag, IntegerType::get(ctx, 8),
                                                cali_type.getExpressedType(),
                                                scale, zeropoint, qmin, qmax);
  return RankedTensorType::get(type.getShape(), qtype);
}

void PoolMaskLowering::LoweringINT8(PatternRewriter &rewriter,
                                    top::PoolMaskOp op, bool asymmetric) const {
  std::vector<NamedAttribute> attrs;
  for (auto &attr : op->getAttrs()) {
    attrs.emplace_back(attr);
  }
  auto mask_shape = module::getShape(op.getOutput());
  auto output_type = op.getOutput().getType().cast<RankedTensorType>();
  auto quant_type = quant::CalibratedQuantizedType::get(
      output_type.getElementType(), -127, 127);
  auto new_type = RankedTensorType::get(mask_shape, quant_type);
  auto pool_mask_type = createQuantInt8Type(new_type);

  rewriter.replaceOpWithNewOp<tpu::PoolMaskOp>(
      op, pool_mask_type, ValueRange{op.getInput()}, attrs);
}

void PoolMaskLowering::LoweringBF16(PatternRewriter &rewriter,
                                    top::PoolMaskOp op) const {
  lowering_common_bf16<tpu::PoolMaskOp>(rewriter, op);
}

} // namespace cv18xx
} // namespace tpu_mlir
