//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Dialect/Tpu/IR/TpuOps.h"
#include "tpu_mlir/Dialect/Tpu/Transforms/BM168x/DoExtraOpt.h"
#include "tpu_mlir/Support/Module.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Format.h"

#include <fstream>
#include <set>
#include <sstream>

using namespace llvm;

namespace tpu_mlir {
namespace bm1684x {

LogicalResult
MatMulHdimBatchPattern::matchAndRewrite(tpu::MatMulOp op,
                                        PatternRewriter &rewriter) const {

//  if (module::isAsymmetric()) {
//    return failure();
//  }

  auto left = op.getInput();
  auto right = op.getRight();
  auto out = op.getResult();

  auto stype = module::getStorageType(left);
  if (stype.isF32()) {
    return failure();
  }

  if (auto l_trans_op = dyn_cast<tpu::PermuteOp>(left.getDefiningOp())) {
    if (!l_trans_op->hasOneUse()) {
      return failure();
    }
    auto r_trans_op = dyn_cast<tpu::PermuteOp>(right.getDefiningOp());
    if (!(r_trans_op && r_trans_op->hasOneUse())) {
      return failure();
    }

    auto l_order = module::getI64Array(l_trans_op.getOrder());
    auto r_order = module::getI64Array(r_trans_op.getOrder());
    if (l_order->size() == 4 && l_order->at(0) == 0 && l_order->at(1) == 2 &&
        r_order->size() == 4 && r_order->at(0) == 0 && r_order->at(1) == 2) {
      auto hdim_is_batch = op.getHdimIsBatch();
      op->setAttr("hdim_is_batch", rewriter.getBoolAttr(!hdim_is_batch));
      if (l_order->at(2) == 3 && l_order->at(3) == 1) {
        auto l_trans = op.getLeftTranspose();
        op->setAttr("left_transpose", rewriter.getBoolAttr(!l_trans));
      }
      if (r_order->at(2) == 3 && r_order->at(3) == 1) {
        auto r_trans = op.getRightTranspose();
        op->setAttr("right_transpose", rewriter.getBoolAttr(!r_trans));
      }
      op->setOperand(0, l_trans_op.getInput());
      op->setOperand(1, r_trans_op.getInput());
      rewriter.eraseOp(l_trans_op);
      rewriter.eraseOp(r_trans_op);
      // modify matmul out shape and name
      auto mat_out = op->getResult(0);
      auto trans_type = mat_out.getType();
      auto out_shape = module::getShape(mat_out);
      std::vector<int64_t> new_out_shape(4, 0);
      new_out_shape[0] = out_shape[0];
      new_out_shape[1] = out_shape[2];
      new_out_shape[2] = out_shape[1];
      new_out_shape[3] = out_shape[3];
      auto new_out_type =
          RankedTensorType::get(new_out_shape, module::getElementType(mat_out));
      mat_out.setType(new_out_type);
      auto out_name = module::getName(mat_out).str();
      auto new_loc =
          NameLoc::get(rewriter.getStringAttr(out_name + "_hdim_is_batch"));
      op->setLoc(new_loc);

      // Add Transpose(0,2,1,3) to output
      rewriter.setInsertionPointAfter(op);
      std::vector<NamedAttribute> attrs;
      std::vector<int64_t> out_order = {0, 2, 1, 3};
      attrs.push_back(
          rewriter.getNamedAttr("order", rewriter.getI64ArrayAttr(out_order)));
      auto trans_loc = NameLoc::get(rewriter.getStringAttr(out_name));
      auto trans_op = rewriter.create<tpu::PermuteOp>(
          trans_loc, trans_type, ValueRange{mat_out, module::getNoneOp(op)},
          attrs);
      rewriter.replaceAllUsesExcept(mat_out, trans_op->getResult(0), trans_op);

      return success();
    }
  }
  return failure();
}

LogicalResult
MatMulLeftReusePattern::matchAndRewrite(tpu::MatMulOp op,
                                        PatternRewriter &rewriter) const {
  auto in_op = op.getInput().getDefiningOp();
  if (in_op->hasOneUse()) {
    op.setLeftReuse(0);
  } else {
    op.setLeftReuse(1);
  }
  return success();
}

} // namespace bm1684x
} // namespace tpu_mlir
