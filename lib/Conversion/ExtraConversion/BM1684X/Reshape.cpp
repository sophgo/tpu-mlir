//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Conversion/ExtraConversion/ExtraConvertBM1684X.h"

using namespace llvm;

namespace tpu_mlir {
namespace bm1684x {

// reorder op when reshapeOp is before matmul/mulconst/cast/softmax op to
// eliminate reshapeOp
LogicalResult
ReshapeReorderPattern::matchAndRewrite(top::ReshapeOp op,
                                       PatternRewriter &rewriter) const {
  auto output = op.getOutput();
  if (!output.hasOneUse()) {
    return failure();
  }
  auto next_op_ = *output.getUsers().begin();

  if (auto next_op = dyn_cast<top::MatMulOp>(next_op_)) {
    // right is from Reshape too
    auto left = next_op.getInput();
    auto right = next_op.getRight();
    auto right_op_ = right.getDefiningOp();
    auto right_op = dyn_cast<top::ReshapeOp>(right_op_);
    if (op != left.getDefiningOp() || !right_op) {
      return failure();
    }
    // check left and right are both Reshape(n, c, h, w) --> (nxc, h, w)
    auto lshape_ = SmallVector<int64_t>(module::getShape(op.getInput()));
    auto lshape = module::getShape(left);
    if (!(lshape.size() == 3 && lshape_.size() == 4 &&
          lshape[0] == lshape_[0] * lshape_[1] && lshape[1] == lshape_[2] &&
          lshape[2] == lshape_[3])) {
      return failure();
    }
    auto rshape_ = module::getShape(right_op.getInput());
    auto rshape = SmallVector<int64_t>(module::getShape(right));
    if (!(rshape.size() == 3 && rshape_.size() == 4 &&
          rshape[0] == rshape_[0] * rshape_[1] && rshape[1] == rshape_[2] &&
          rshape[2] == rshape_[3])) {
      return failure();
    }
    if (lshape_[0] != rshape_[0] || lshape_[1] != rshape_[1]) {
      return failure();
    }

    // remove left and right ReshapeOp
    op.replaceAllUsesWith(op.getInput());
    right_op.replaceAllUsesWith(right_op.getInput());

    // Update MatMul output shape
    // and update loc to avoid comparing
    auto next_out = next_op.getOutput();
    auto ori_out_type = next_out.getType();
    auto oshape = module::getShape(next_out);
    std::vector<int64_t> new_oshape{lshape_[0], lshape_[1], oshape[1],
                                    oshape[2]};
    auto new_out_type =
        RankedTensorType::get(new_oshape, module::getElementType(next_out));
    next_out.setType(new_out_type);
    auto ori_name = module::getName(next_out).str();
    auto new_loc = NameLoc::get(rewriter.getStringAttr(ori_name + "_Reshape"));
    next_op->setLoc(new_loc);

    // Add ReshapeOp after MatMul
    rewriter.setInsertionPointAfterValue(next_out);
    auto ori_loc = NameLoc::get(rewriter.getStringAttr(ori_name));
    auto new_reshape_op =
        rewriter.create<top::ReshapeOp>(ori_loc, ori_out_type, ValueRange{next_out});
    next_out.replaceAllUsesExcept(new_reshape_op.getOutput(), new_reshape_op);

    return success();
  } else if (isa<top::MulConstOp, top::CastOp, top::SoftmaxOp>(next_op_)) {
    // check input is Reshape(n, c, h, w) --> (nxc, h, w)
    auto ishape = SmallVector<int64_t>(module::getShape(op.getInput()));
    auto next_ishape = module::getShape(op.getOutput());
    if (!(next_ishape.size() == 3 && ishape.size() == 4 &&
          next_ishape[0] == ishape[0] * ishape[1] &&
          next_ishape[1] == ishape[2] && next_ishape[2] == ishape[3])) {
      return failure();
    }
    // check next_op param
    if (auto next_op = dyn_cast<top::SoftmaxOp>(next_op_)) {
      int64_t axis = next_op.getAxis();
      if (axis != 2 || axis == -1) {
        return failure();
      }
    }

    // remove ReshapeOp
    op.replaceAllUsesWith(op.getInput());

    // update next_op output shape and modify loc name to avoid comparing
    auto next_out = next_op_->getResult(0);
    auto ori_out_type = next_out.getType();
    auto new_out_type =
        RankedTensorType::get(ishape, module::getElementType(next_out));
    next_out.setType(new_out_type);
    auto ori_name = module::getName(next_out).str();
    auto new_loc = NameLoc::get(rewriter.getStringAttr(ori_name + "_Reshape"));
    next_op_->setLoc(new_loc);

    // Add ReshapeOp after MulConst/Cast/Softmax
    rewriter.setInsertionPointAfterValue(next_out);
    auto ori_loc = NameLoc::get(rewriter.getStringAttr(ori_name));
    auto new_reshape_op =
        rewriter.create<top::ReshapeOp>(ori_loc, ori_out_type, ValueRange{next_out});
    next_out.replaceAllUsesExcept(new_reshape_op.getOutput(), new_reshape_op);

    if (auto next_op = dyn_cast<top::SoftmaxOp>(next_op_)) {
      next_op->setAttr("axis", rewriter.getSI32IntegerAttr(3));
    }

    return success();
  } else if (auto next_op = dyn_cast<top::ReshapeOp>(next_op_)) {
    auto ishape = module::getShape(op.getInput());
    auto next_oshape = module::getShape(next_op.getOutput());
    if (ishape != next_oshape) {
      return failure();
    }

    op.replaceAllUsesWith(op.getInput());
    next_op.replaceAllUsesWith(next_op.getInput());
    return success();
  }

  return failure();
}

} // namespace bm1684x
} // namespace tpu_mlir
