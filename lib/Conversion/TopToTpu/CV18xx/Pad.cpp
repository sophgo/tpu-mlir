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
  auto in_thr = module::getThreshold(op.input());
  auto in_scale = module::getScale(in_thr, true);
  std::vector<NamedAttribute> attrs;
  auto val = op.val().convertToDouble();
  val = to_int8(val / in_scale, ROUNDING_HALF_UP);
  attrs.push_back(rewriter.getNamedAttr("paddings", op.paddingsAttr()));
  attrs.push_back(rewriter.getNamedAttr("val", rewriter.getF64FloatAttr(val)));
  attrs.push_back(rewriter.getNamedAttr("mode", op.modeAttr()));
  std::vector<Value> operands;
  operands.push_back(op.input());
  if (op.mode() == 1) {
    // pad reflect
   auto nofDims = module::getShape(op.input()).size();
    auto pads = module::getI64Array(op.paddings());
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
  auto newType = getQuantInt8Type(op.output(), asymmetric);
  rewriter.replaceOpWithNewOp<tpu::PadOp>(op, newType, operands, attrs);
}

void PadLowering::LoweringBF16(PatternRewriter &rewriter, top::PadOp op) const {
  std::vector<Value> operands;
  operands.push_back(op.input());
  if (op.mode() == 1) {
    // pad reflect
    auto nofDims = module::getShape(op.input()).size();
    auto pads = module::getI64Array(op.paddings());
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
  auto newType = getQuantBF16Type(op->getResult(0));
  rewriter.replaceOpWithNewOp<tpu::PadOp>(op, newType, operands,
                                          op->getAttrs());
}
} // namespace cv18xx
} // namespace tpu_mlir
