//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Dialect/Tpu/IR/TpuOps.h"
#include "tpu_mlir/Support/Module.h"
#include "tpu_mlir/Backend/Arch.h"


using namespace llvm;
namespace tpu_mlir {

namespace bm1684x {
class MatMulHdimBatchPattern : public OpRewritePattern<tpu::MatMulOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(tpu::MatMulOp op,
                                PatternRewriter &rewriter) const override {

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
    auto l_trans_op = dyn_cast<tpu::PermuteOp>(left.getDefiningOp());
    if (!(l_trans_op && l_trans_op->hasOneUse())) {
      return failure();
    }
    auto r_trans_op = dyn_cast<tpu::PermuteOp>(right.getDefiningOp());
    if (!(r_trans_op && r_trans_op->hasOneUse())) {
      return failure();
    }

    auto l_order = module::getI64Array(l_trans_op.getOrder());
    auto r_order = module::getI64Array(r_trans_op.getOrder());
    if (false ==
        (l_order->size() == 4 && l_order->at(0) == 0 && l_order->at(1) == 2 &&
         r_order->size() == 4 && r_order->at(0) == 0 && r_order->at(1) == 2)) {
      return failure();
    }
    auto l_trans = op.getLeftTranspose();
    auto r_trans = op.getRightTranspose();
    if (l_order->at(2) == 3 && l_order->at(3) == 1) {
      l_trans = !l_trans;
    }
    if (r_order->at(2) == 3 && r_order->at(3) == 1) {
      r_trans = !r_trans;
    }
    if (l_trans == true && r_trans == false) {
      // mm2 not support l_trans && !r_trans
      return failure();
    }
    auto hdim_is_batch = op.getHdimIsBatch();
    op->setAttr("hdim_is_batch", rewriter.getBoolAttr(!hdim_is_batch));
    op->setAttr("left_transpose", rewriter.getBoolAttr(l_trans));
    op->setAttr("right_transpose", rewriter.getBoolAttr(r_trans));
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
};

class MatMulLeftReusePattern : public OpRewritePattern<tpu::MatMulOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(tpu::MatMulOp op,
                                PatternRewriter &rewriter) const override {
    auto in_op = op.getInput().getDefiningOp();
    if (in_op->hasOneUse()) {
      op.setLeftReuse(0);
    } else {
      op.setLeftReuse(1);
    }
    return success();
  }
};

// reorder op when transpose is before mulconst/cast/softmax to optimize bert
class PermuteReorderPattern : public OpRewritePattern<tpu::PermuteOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(tpu::PermuteOp op,
                                PatternRewriter &rewriter) const override {
    //  if (module::isAsymmetric()) {
    //    return failure();
    //  }

    if (op->hasOneUse() == false) {
      return failure();
    }
    std::vector<int64_t> ps = {0, 2, 1, 3};
    auto order = module::getI64Array(op.getOrder());
    if (*order != ps) {
      return failure();
    }

    auto in_shape = module::getShape(op.getInput());
    auto out_shape = module::getShape(op.getOutput());
    auto nextOp = *op.getOutput().getUsers().begin();
    if (nextOp->hasOneUse() == false) {
      return failure();
    }
    if (auto mulconst_op = dyn_cast<tpu::MulConstOp>(nextOp)) {
      auto newType = RankedTensorType::get(
          in_shape, module::getElementType(mulconst_op.getOutput()));
      mulconst_op.getOutput().setType(newType);
      op.replaceAllUsesWith(op.getInput());
      rewriter.setInsertionPointAfter(mulconst_op);
      newType = RankedTensorType::get(
          out_shape, module::getElementType(mulconst_op.getOutput()));
      auto out_loc = mulconst_op.getLoc(); // keep out location unchanged.
      auto name = module::getName(mulconst_op.getOutput());
      auto loc = NameLoc::get(rewriter.getStringAttr(name + "_trans"));
      mulconst_op->setLoc(loc);
      std::vector<NamedAttribute> attrs;
      attrs.push_back(
          rewriter.getNamedAttr("order", rewriter.getI64ArrayAttr(ps)));
      auto new_op = rewriter.create<tpu::PermuteOp>(
          out_loc, newType,
          ValueRange{mulconst_op.getOutput(), module::getNoneOp(mulconst_op)},
          attrs);
      mulconst_op.getOutput().replaceAllUsesExcept(new_op.getOutput(),
                                                   {new_op});
      return success();
    } else if (auto cast_op = dyn_cast<tpu::CastOp>(nextOp)) {
      auto newType = RankedTensorType::get(
          in_shape, module::getElementType(cast_op.getOutput()));
      cast_op.getOutput().setType(newType);
      op.replaceAllUsesWith(op.getInput());
      rewriter.setInsertionPointAfter(cast_op);
      newType = RankedTensorType::get(
          out_shape, module::getElementType(cast_op.getOutput()));
      auto out_loc = cast_op.getLoc(); // keep out location unchanged.
      auto name = module::getName(cast_op.getOutput());
      auto loc = NameLoc::get(rewriter.getStringAttr(name + "_trans"));
      cast_op->setLoc(loc);
      std::vector<NamedAttribute> attrs;
      attrs.push_back(
          rewriter.getNamedAttr("order", rewriter.getI64ArrayAttr(ps)));
      auto new_op = rewriter.create<tpu::PermuteOp>(
          out_loc, newType,
          ValueRange{cast_op.getOutput(), module::getNoneOp(cast_op)}, attrs);
      cast_op.getOutput().replaceAllUsesExcept(new_op.getOutput(), {new_op});
      return success();
    } else if (auto add_op = dyn_cast<tpu::AddOp>(nextOp)) {
      auto inB = add_op.getInputs()[1];
      if (!module::isWeight(inB)) {
        return failure();
      }
      auto inB_shape = module::getShape(inB);
      if (inB_shape[1] != 1) {
        return failure();
      }
      std::vector<int64_t> new_inB_shape = {inB_shape[0], inB_shape[2],
                                            inB_shape[1], inB_shape[3]};
      auto newType =
          RankedTensorType::get(new_inB_shape, module::getElementType(inB));
      inB.setType(newType);

      newType = RankedTensorType::get(
          in_shape, module::getElementType(add_op.getOutput()));
      add_op.getOutput().setType(newType);
      op.replaceAllUsesWith(op.getInput());
      rewriter.setInsertionPointAfter(add_op);
      newType = RankedTensorType::get(
          out_shape, module::getElementType(add_op.getOutput()));
      auto out_loc = add_op.getLoc(); // keep out location unchanged.
      auto name = module::getName(add_op.getOutput());
      auto loc = NameLoc::get(rewriter.getStringAttr(name + "_trans"));
      add_op->setLoc(loc);
      std::vector<NamedAttribute> attrs;
      attrs.push_back(
          rewriter.getNamedAttr("order", rewriter.getI64ArrayAttr(ps)));
      auto new_op = rewriter.create<tpu::PermuteOp>(
          out_loc, newType,
          ValueRange{add_op.getOutput(), module::getNoneOp(add_op)}, attrs);
      add_op.getOutput().replaceAllUsesExcept(new_op.getOutput(), {new_op});
      return success();
    } else if (auto mul_op = dyn_cast<tpu::MulOp>(nextOp)) {
      auto inB = mul_op.getInputs()[1];
      if (!module::isWeight(inB)) {
        return failure();
      }
      auto inB_shape = module::getShape(inB);
      if (inB_shape[1] != 1) {
        return failure();
      }
      std::vector<int64_t> new_inB_shape = {inB_shape[0], inB_shape[2],
                                            inB_shape[1], inB_shape[3]};
      auto newType =
          RankedTensorType::get(new_inB_shape, module::getElementType(inB));
      inB.setType(newType);

      newType = RankedTensorType::get(
          in_shape, module::getElementType(mul_op.getOutput()));
      mul_op.getOutput().setType(newType);
      op.replaceAllUsesWith(op.getInput());
      rewriter.setInsertionPointAfter(mul_op);
      newType = RankedTensorType::get(
          out_shape, module::getElementType(mul_op.getOutput()));
      auto out_loc = mul_op.getLoc(); // keep out location unchanged.
      auto name = module::getName(mul_op.getOutput());
      auto loc = NameLoc::get(rewriter.getStringAttr(name + "_trans"));
      mul_op->setLoc(loc);
      std::vector<NamedAttribute> attrs;
      attrs.push_back(
          rewriter.getNamedAttr("order", rewriter.getI64ArrayAttr(ps)));
      auto new_op = rewriter.create<tpu::PermuteOp>(
          out_loc, newType,
          ValueRange{mul_op.getOutput(), module::getNoneOp(mul_op)}, attrs);
      mul_op.getOutput().replaceAllUsesExcept(new_op.getOutput(), {new_op});
      return success();

    } else if (auto softmax_op = dyn_cast<tpu::SoftmaxOp>(nextOp)) {
      int64_t axis = softmax_op.getAxis();
      if (!(axis == -1 || axis == out_shape.size() - 1)) {
        return failure();
      }
      auto newType = RankedTensorType::get(
          in_shape, module::getElementType(softmax_op.getOutput()));
      softmax_op.getOutput().setType(newType);
      op.replaceAllUsesWith(op.getInput());
      rewriter.setInsertionPointAfter(softmax_op);
      newType = RankedTensorType::get(
          out_shape, module::getElementType(softmax_op.getOutput()));
      auto out_loc = softmax_op.getLoc(); // keep out location unchanged.
      auto name = module::getName(softmax_op.getOutput());
      auto loc = NameLoc::get(rewriter.getStringAttr(name + "_trans"));
      softmax_op->setLoc(loc);
      std::vector<NamedAttribute> attrs;
      attrs.push_back(
          rewriter.getNamedAttr("order", rewriter.getI64ArrayAttr(ps)));
      auto new_op = rewriter.create<tpu::PermuteOp>(
          out_loc, newType,
          ValueRange{softmax_op.getOutput(), module::getNoneOp(softmax_op)},
          attrs);
      softmax_op.getOutput().replaceAllUsesExcept(new_op.getOutput(), {new_op});
      return success();
    } else if (auto permute_op = dyn_cast<tpu::PermuteOp>(nextOp)) {
      auto next_order = module::getI64Array(op.getOrder());
      if (*next_order != ps) {
        return failure();
      }
      permute_op.replaceAllUsesWith(op.getInput());
      // op.replaceAllUsesWith(op.geInput());
      rewriter.eraseOp(permute_op);
      rewriter.eraseOp(op);
      return success();
    }

    return failure();
  }
};
} // namespace bm1684x

namespace tpu {
using namespace bm1684x;
void populateOptimizeBM1684XPatterns(RewritePatternSet *patterns) {
  // clang-format off
    patterns->add<
      MatMulHdimBatchPattern,
      MatMulLeftReusePattern,
      PermuteReorderPattern
    >(patterns->getContext());
  // clang-format on
}
} // namespace tpu

} // namespace tpu_mlir
