//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Conversion/TopToTpu/LoweringBM1684X.h"

namespace tpu_mlir {
namespace bm1684x {

static void set_hswish_attr(PatternRewriter &rewriter, top::HardSwishOp op) {
  auto op_ = op.getOperation();
  op_->setAttr("mode", tpu::ActiveModeAttr::get(op.getContext(),
                                                tpu::ActiveMode::HSWISH));
}

void HardSwishLowering::LoweringF32(PatternRewriter &rewriter,
                                    top::HardSwishOp op) const {
  set_hswish_attr(rewriter, op);
  lowering_common_f32<tpu::ActiveOp>(rewriter, op);
}

static inline double hswish(double x) {
  return x * std::max(0.0, std::min(1.0, x / 6 + 0.5));
}
void HardSwishLowering::LoweringINT4(PatternRewriter &rewriter, top::HardSwishOp op,
                                   bool asymmetric) const {
  LoweringINT8(rewriter, op, asymmetric);
}
void HardSwishLowering::LoweringINT8(PatternRewriter &rewriter,
                                     top::HardSwishOp op,
                                     bool asymmetric) const {
  auto stype = module::getStorageType(op.output());
  Value table = create_lookup_table(op.input(), op.output(), asymmetric,
                                    [](double val) { return hswish(val); });
  auto newType = getQuantInt8Type(op.output(), asymmetric);
  rewriter.replaceOpWithNewOp<tpu::LutOp>(op, newType,
                                          ValueRange{op.input(), table});
}

void HardSwishLowering::LoweringBF16(PatternRewriter &rewriter,
                                     top::HardSwishOp op) const {
  set_hswish_attr(rewriter, op);
  lowering_common_bf16<tpu::ActiveOp>(rewriter, op);
}

void HardSwishLowering::LoweringF16(PatternRewriter &rewriter,
                                    top::HardSwishOp op) const {
  set_hswish_attr(rewriter, op);
  lowering_common_f16<tpu::ActiveOp>(rewriter, op);
}

void HardSwishLowering::LoweringQuantized(PatternRewriter &rewriter,
                                          top::HardSwishOp op) const {
  llvm_unreachable("Not Implemented");
}

} // namespace bm1684x
} // namespace tpu_mlir
