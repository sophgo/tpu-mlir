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
void PadLowering::LoweringINT8(PatternRewriter &rewriter, top::PadOp op,
                               bool asymmetric) const {
  auto in_thr = module::getThreshold(op.getInput());
  auto in_scale = module::getScale(in_thr, true);
  std::vector<NamedAttribute> attrs;
  auto val = op.getVal().convertToDouble();
  val = to_int8(val / in_scale, ROUNDING_HALF_UP);
  attrs.push_back(rewriter.getNamedAttr("paddings", op.getPaddingsAttr()));
  attrs.push_back(rewriter.getNamedAttr("val", rewriter.getF64FloatAttr(val)));
  auto m = tpu::symbolizePaddingMode(op.getMode()).value_or(tpu::PaddingMode::constant);
  attrs.push_back(rewriter.getNamedAttr(
      "mode", tpu::PaddingModeAttr::get(op->getContext(), m)));
  std::vector<Value> operands;
  operands.push_back(op.getInput());
  if (m == tpu::PaddingMode::reflect) {
    // pad reflect
    auto nofDims = module::getShape(op.getInput()).size();
    auto pads = module::getI64Array(op.getPaddings());
    int32_t count = 0;
    for (int i = 0; i < pads->size(); i++) {
      if (pads->at(i) != 0) {
        ++count;
      }
    }
    assert(count <= 4 && "only support 1D / 2D reflectionpad!");
    std::vector<int64_t> lrpad = {pads->at(nofDims - 1),
                                  pads->at(nofDims * 2 - 1)};
    for (int i = 0; i < lrpad.size(); i++) {
      auto k = lrpad.at(i);
      if(k == 0) {
        operands.push_back(module::getNoneOp(op));
        continue;
      }
      auto select = std::vector<int8_t>(k * k, 0);
      for (int i = 0; i < k; i++) {
        int last = k - i - 1;
        select[i * k + last] = 1;
      }
      auto shape = std::vector<int64_t>{k, k};
      auto type =
          RankedTensorType::get(shape, rewriter.getIntegerType(8, true));
      auto selectOp = top::WeightOp::create(op, "select_" + std::to_string(i),
                                            select, type);
      operands.push_back(selectOp);
    }
  } else {
    operands.push_back(module::getNoneOp(op));
    operands.push_back(module::getNoneOp(op));
  }
  operands.push_back(module::getNoneOp(op));
  auto newType = getQuantInt8Type(op.getOutput(), asymmetric);
  rewriter.replaceOpWithNewOp<tpu::PadOp>(op, newType, operands, attrs);
}

void PadLowering::LoweringBF16(PatternRewriter &rewriter, top::PadOp op) const {
  auto m = tpu::symbolizePaddingMode(op.getMode())
               .value_or(tpu::PaddingMode::constant);
  auto op_ = op.getOperation();
  op_->setAttr("mode", tpu::PaddingModeAttr::get(op.getContext(), m));
  std::vector<Value> operands;
  operands.push_back(op.getInput());
  if (m == tpu::PaddingMode::reflect) {
    // pad reflect
    auto nofDims = module::getShape(op.getInput()).size();
    auto pads = module::getI64Array(op.getPaddings());
    int32_t count = 0;
    for (int i = 0; i < pads->size(); i++) {
      if (pads->at(i) != 0) {
        ++count;
      }
    }
    assert(count <= 4 && "only support 1D / 2D reflectionpad!");
    std::vector<int64_t> lrpad = {pads->at(nofDims - 1),
                                  pads->at(nofDims * 2 - 1)};
    for (int i = 0; i < lrpad.size(); i++) {
      auto k = lrpad.at(i);
      if(k == 0) {
        operands.push_back(module::getNoneOp(op));
        continue;
      }
      auto select = std::vector<float_t>(k * k, 0);
      for (int i = 0; i < k; i++) {
        int last = k - i - 1;
        select[i * k + last] = 1;
      }
      auto shape = std::vector<int64_t>{k, k};
      auto type =
          // RankedTensorType::get(shape, rewriter.getIntegerType(8, true));
          RankedTensorType::get(shape, rewriter.getF32Type());
      auto v = top::WeightOp::create(op, "select_" + std::to_string(i), select,
                                     type);
      auto selectOp = cast<top::WeightOp>(v.getDefiningOp());
      operands.push_back(selectOp.clone_bf16(op));
    }
  } else {
    operands.push_back(module::getNoneOp(op));
    operands.push_back(module::getNoneOp(op));
  }
  operands.push_back(module::getNoneOp(op));
  auto newType = getQuantBF16Type(op->getResult(0));
  rewriter.replaceOpWithNewOp<tpu::PadOp>(op, newType, operands,
                                          op->getAttrs());
}
} // namespace cv18xx
} // namespace tpu_mlir
