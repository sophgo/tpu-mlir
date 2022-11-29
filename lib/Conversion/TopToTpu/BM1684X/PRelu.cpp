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

void PReluLowering::LoweringINT8(PatternRewriter &rewriter, top::PReluOp op,
                                 bool asymmetric) const {
  if (asymmetric == false) {
    int64_t N, C, H, W;
    Module::getNCHW(op.output(), N, C, H, W);
    auto src_shape = Module::getShape(op.input());
    auto slope_shape = Module::getShape(op.slope());
    auto num_slope = Module::getNumElements(op.slope());
    assert(num_slope == C);

    std::vector<Value> operands;
    std::vector<NamedAttribute> attrs;

    auto slopeOp = cast<top::WeightOp>(op.slope().getDefiningOp());
    auto slope_f32 = slopeOp.read<float>();
    std::vector<int8_t> slope_int8(num_slope);
    auto scale_max = std::abs(slope_f32->at(0));
    for (int idx = 1; idx < num_slope; idx++) {
      auto scale = std::abs(slope_f32->at(idx));
      if (scale > scale_max) {
        scale_max = scale;
      }
    }
    int64_t scalei, rshifti;
    getRShiftAndMultiplierFromQScale(scale_max, &scalei, &rshifti);

    for (int idx = 0; idx < num_slope; idx++) {
      slope_int8[idx] =
          getMultiplierI8FromQScaleAndRShift(slope_f32->at(idx), rshifti);
    }

    operands.push_back(op.input());
    auto new_type = RankedTensorType::get(slope_shape, rewriter.getI8Type());
    auto new_slope =
        top::WeightOp::create(op, "slope_i8", slope_int8, new_type);
    operands.push_back(new_slope);

    for (auto &attr : op->getAttrs()) {
      attrs.push_back(attr);
    }
    attrs.push_back(
        rewriter.getNamedAttr("rshift", rewriter.getSI32IntegerAttr(rshifti)));
    auto newType = Quant::getQuantInt8Type(op.output(), asymmetric);
    rewriter.replaceOpWithNewOp<tpu::PReluOp>(op, newType, operands, attrs);
  } else {
    LoweringF32(rewriter, op);
  }
}

void PReluLowering::LoweringF32(PatternRewriter &rewriter,
                                top::PReluOp op) const {
  rewriter.setInsertionPointAfter(op);
  std::vector<Value> operands;
  const int nInputs = op->getNumOperands();
  for (auto i = 0; i < nInputs; ++i) {
    operands.push_back(op->getOperand(i));
  }

  auto src_shape = Module::getShape(op.input());
  auto slope_shape = Module::getShape(op.slope());
  int src_dims = src_shape.size();
  int slope_dims = slope_shape.size();
  assert(src_dims == slope_dims);

  bool channel_share = false;
  if (slope_shape[1] == 1) {
    channel_share = true;
  } else {
    for (int i = 0; i < slope_dims; i++) {
      if (i != 1 && slope_shape[i] != 1) {
        assert(0);
      }
    }
  }

  auto slopeOp = cast<top::WeightOp>(op.slope().getDefiningOp());
  auto slope_f32 = slopeOp.read<float>();

  std::vector<NamedAttribute> attrs;
  for (auto &attr : op->getAttrs()) {
    attrs.push_back(attr);
  }
  attrs.push_back(rewriter.getNamedAttr("channel_shared",
                                        rewriter.getBoolAttr(channel_share)));
  if (channel_share) {
    float *slope_data = slope_f32->data();
    float slope_val = (float)(*slope_data);
    attrs.push_back(rewriter.getNamedAttr("slope_val",
                                          rewriter.getF64FloatAttr(slope_val)));
  }
  Value newValue;
  auto newOp = rewriter.create<tpu::PReluOp>(
      op->getLoc(), op.output().getType(), operands, attrs);
  newValue = newOp.output();
  rewriter.replaceOp(op, {newValue});
}

void PReluLowering::LoweringBF16(PatternRewriter &rewriter,
                                 top::PReluOp op) const {
  rewriter.setInsertionPointAfter(op);

  auto src_shape = Module::getShape(op.input());
  auto slope_shape = Module::getShape(op.slope());

  std::vector<Value> operands;
  auto slopeOp = cast<top::WeightOp>(op.slope().getDefiningOp());
  operands.push_back(op.input());

  std::vector<NamedAttribute> attrs;
  for (auto &attr : op->getAttrs()) {
    attrs.push_back(attr);
  }
  operands.push_back(slopeOp.clone_bf16(op));
  auto newType = getQuantBF16Type(op.output());
  rewriter.replaceOpWithNewOp<tpu::PReluOp>(op, newType, operands, attrs);
}

void PReluLowering::LoweringF16(PatternRewriter &rewriter,
                                top::PReluOp op) const {
  rewriter.setInsertionPointAfter(op);

  auto src_shape = Module::getShape(op.input());
  auto slope_shape = Module::getShape(op.slope());

  std::vector<Value> operands;
  auto slopeOp = cast<top::WeightOp>(op.slope().getDefiningOp());
  operands.push_back(op.input());

  std::vector<NamedAttribute> attrs;
  for (auto &attr : op->getAttrs()) {
    attrs.push_back(attr);
  }
  operands.push_back(slopeOp.clone_f16(op));
  auto newType = getQuantF16Type(op.output());
  rewriter.replaceOpWithNewOp<tpu::PReluOp>(op, newType, operands, attrs);
}

void PReluLowering::LoweringQuantized(PatternRewriter &rewriter,
                                      top::PReluOp op) const {
  llvm_unreachable("Not Implemented");
}

} // namespace bm1684x
} // namespace tpu_mlir
