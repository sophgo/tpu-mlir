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

static bool slice_fusible(tpu::SliceOp op) {
  std::vector<int64_t> i_s;
  std::vector<int64_t> o_s;
  std::vector<int> offset_4;
  std::vector<int> step_4;
  bool fusible = false;
  op.parseParam(i_s, o_s, offset_4, step_4, fusible);
  return fusible;
}

static bool is_fusible_op(Operation *op) {
  if (auto concatOp = dyn_cast<tpu::ConcatOp>(op)) {
    return !concatOp.only_merge();
  }
  if (isa<tpu::ReshapeOp>(op)) {
    return false;
  }
  if (auto sliceOp = dyn_cast<tpu::SliceOp>(op)) {
    return slice_fusible(sliceOp);
  }
  return true;
}

static bool fusible(top::ConcatOp concatOp) {
  bool only_merge = !concatOp.do_relu();
  // check concatOp's outer_dim
  if (only_merge) {
    auto shape = Module::getShape(concatOp.output());
    int outer_dim = std::accumulate(shape.begin(), shape.begin() + concatOp.axis(), 1,
                                    std::multiplies<int64_t>());
    if (outer_dim != 1) {
      return false;
    }
  }
  // check concatOp's input
  uint32_t nInputs = concatOp->getNumOperands();
  for (int i = 0; i < nInputs; ++i) {
    if (only_merge == false) {
      return false;
    }
    auto inOp = concatOp->getOperand(i).getDefiningOp();
    if (isa<top::WeightOp>(inOp)) {
      return false;
    }
    only_merge = is_fusible_op(inOp);
    if (only_merge && !inOp->getResult(0).hasOneUse()) {
      for (auto &use : inOp->getResult(0).getUses()){
        auto useOp = use.getOwner();
        if (!is_fusible_op(useOp)) {
          return false;
        }
      }
    }
  }
  return only_merge;
}

void ConcatLowering::LoweringINT8(PatternRewriter &rewriter,
                                  top::ConcatOp concatOp,
                                  bool asymmetric) const {
  std::vector<Value> operands;
  double out_thr = Quant::getThreshold(concatOp.output());
  uint32_t nInputs = concatOp->getNumOperands();
  bool only_merge = fusible(concatOp);
  auto rshift_v = std::make_unique<std::vector<int64_t>>(nInputs, 0);
  auto multiplier_v = std::make_unique<std::vector<int64_t>>(nInputs, 1);
  for (int i = 0; i < nInputs; ++i) {
    auto in = concatOp->getOperand(i);
    operands.push_back(in);
    if (isa<top::WeightOp>(in.getDefiningOp())) {
      // not test now
      LoweringBF16(rewriter, concatOp);
      return;
    }
    double in_thr = Quant::getThreshold(in);
    double qscale = in_thr / out_thr;
    if (fabs(in_thr - out_thr) <= 1e-5) {
      qscale = 1.0;
    }
    if (qscale != 1.0f) {
      int64_t multiplier = 0;
      int64_t shift = 0;
      getRShiftAndMultiplierFromQScale(qscale, &multiplier, &shift, false);
      rshift_v->at(i) = shift;
      multiplier_v->at(i) = multiplier;
      only_merge = false;
    }
  } // end for
  std::vector<NamedAttribute> attrs;
  for (auto &attr : concatOp->getAttrs()) {
    attrs.push_back(attr);
  }
  attrs.push_back(rewriter.getNamedAttr(
      "multipliers", rewriter.getI64ArrayAttr(*multiplier_v)));
  attrs.push_back(
      rewriter.getNamedAttr("rshifts", rewriter.getI64ArrayAttr(*rshift_v)));
  attrs.push_back(
      rewriter.getNamedAttr("only_merge", rewriter.getBoolAttr(only_merge)));
  auto newType = Quant::getQuantInt8Type(concatOp.output(), asymmetric);
  rewriter.replaceOpWithNewOp<tpu::ConcatOp>(concatOp, newType, operands,
                                             attrs);
}

void ConcatLowering::LoweringBF16(PatternRewriter &rewriter,
                                  top::ConcatOp concatOp) const {
  rewriter.setInsertionPointAfter(concatOp);
  std::vector<Value> operands;
  // auto op = concatOp.getOperation();
  bool only_merge = fusible(concatOp);
  uint32_t nInputs = concatOp->getNumOperands();
  for (int i = 0; i < nInputs; ++i) {
    if (auto constOp =
            dyn_cast<top::WeightOp>(concatOp->getOperand(i).getDefiningOp())) {
      // concatOp.setOperand(i, constOp.clone_bf16(concatOp));
      operands.push_back(constOp.clone_bf16(concatOp));
      continue;
    }
    operands.push_back(concatOp->getOperand(i));
  }
  std::vector<NamedAttribute> attrs;
  for (auto &attr : concatOp->getAttrs()) {
    attrs.push_back(attr);
  }
  attrs.push_back(rewriter.getNamedAttr("only_merge", rewriter.getBoolAttr(only_merge)));

  auto newType = getQuantBF16Type(concatOp.output());
  auto newOp =
        rewriter.create<tpu::ConcatOp>(concatOp->getLoc(), newType, operands, attrs);
  rewriter.replaceOp(concatOp, {newOp.output()});
}
} // namespace cv18xx
} // namespace tpu_mlir
