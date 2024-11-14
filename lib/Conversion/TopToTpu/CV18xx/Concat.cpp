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

static bool is_fusible_op(Operation *op) {
  if (auto concatOp = dyn_cast<tpu::ConcatOp>(op)) {
    return !concatOp.getOnlyMerge();
  }
  if (isa<tpu::ReshapeOp>(op) || isa<top::ReshapeOp>(op)) {
    return false;
  }
  if (auto sliceOp = dyn_cast<tpu::SliceOp>(op)) {
    // auto p = sliceOp.parseParam();
    // return p.fusible;
    return false;
  }
  if (auto sliceOp = dyn_cast<top::SliceOp>(op)) {
    return false;
  }
  return true;
}

static bool fusible(top::ConcatOp concatOp) {
  bool only_merge = !concatOp.getDoRelu();
  // check concatOp's outer_dim
  if (only_merge) {
    auto shape = module::getShape(concatOp.getOutput());
    int outer_dim =
        std::accumulate(shape.begin(), shape.begin() + concatOp.getAxis(), 1,
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
      for (auto &use : inOp->getResult(0).getUses()) {
        auto useOp = use.getOwner();
        if (!is_fusible_op(useOp)) {
          return false;
        }
      }
    }
  }
  // check next Op
  if (only_merge) {
    for (auto &use : concatOp->getResult(0).getUses()) {
      auto useOp = use.getOwner();
      if (auto concatOp = dyn_cast<tpu::ConcatOp>(useOp)) {
        if (concatOp.getOnlyMerge()) {
          return false;
        }
      }
    }
  }

  // if has same input, only_merge should be false
  if (only_merge) {
    llvm::DenseSet<Value> s;
    for (int i = 0; i < nInputs; i++) {
      auto opd = concatOp.getOperand(i);
      if (s.count(opd) == 1) {
        only_merge = false;
        break;
      } else {
        s.insert(opd);
      }
    }
  }
  return only_merge;
}

void ConcatLowering::LoweringINT8(PatternRewriter &rewriter,
                                  top::ConcatOp concatOp,
                                  bool asymmetric) const {
  std::vector<Value> operands;
  double out_thr = module::getThreshold(concatOp.getOutput());
  uint32_t nInputs = concatOp->getNumOperands();
  bool only_merge = fusible(concatOp);
  auto rshift_v = std::make_unique<std::vector<int64_t>>(nInputs, 0);
  auto multiplier_v = std::make_unique<std::vector<int64_t>>(nInputs, 1);
  for (int i = 0; i < nInputs; ++i) {
    auto in = concatOp->getOperand(i);
    operands.push_back(in);
    if (module::isWeight(in)) {
      // not test now
      LoweringBF16(rewriter, concatOp);
      return;
    }
    double in_thr = module::getThreshold(in);
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
  auto newType = getQuantInt8Type(concatOp.getOutput(), asymmetric);
  rewriter.replaceOpWithNewOp<tpu::ConcatOp>(concatOp, newType, operands,
                                             attrs);
}

void ConcatLowering::LoweringBF16(PatternRewriter &rewriter,
                                  top::ConcatOp concatOp) const {
  auto op = concatOp.getOperation();
  bool only_merge = fusible(concatOp);
  op->setAttr("only_merge", rewriter.getBoolAttr(only_merge));
  lowering_common_bf16<tpu::ConcatOp>(rewriter, op);
}
} // namespace cv18xx
} // namespace tpu_mlir
