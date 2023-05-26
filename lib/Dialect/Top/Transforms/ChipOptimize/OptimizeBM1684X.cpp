//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Support/MathUtils.h"
#include "tpu_mlir/Support/Module.h"
#include "tpu_mlir/Dialect/Top/Transforms/Passes.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace tpu_mlir {

namespace bm1684x {
class ConvertMatMulWithRightTranspose : public OpRewritePattern<top::MatMulOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(top::MatMulOp op,
                                PatternRewriter &rewriter) const override {
    auto filter = op.getRight();
    if (module::isWeight(filter)) {
      return failure();
    }
    if (filter.hasOneUse() == false) {
      return failure();
    }
    auto trans_op = dyn_cast<top::PermuteOp>(filter.getDefiningOp());
    if (!trans_op) {
      return failure();
    }
    auto attr = op.parseParam();
    int to_dim = 2;
    if (attr.batch > 1) {
      to_dim = 3;
    }
    std::vector<int64_t> shape = module::getShape(trans_op.getInput());
    auto order = module::getI64Array(trans_op.getOrder());
    std::vector<int64_t> shape_fix;
    std::vector<int64_t> order_fix;
    auto ret = permute_reset(shape, *order, shape_fix, order_fix, to_dim);
    if (ret == false) {
      return failure();
    }
    int n_idx = to_dim - 2;
    int k_idx = to_dim - 1;
    if (shape_fix[n_idx] == attr.N && shape_fix[k_idx] == attr.K &&
        order_fix[n_idx] == k_idx && order_fix[k_idx] == n_idx) {
      // bingo !
      op.setOperand(1, trans_op.getInput());
      op.setRightTranspose(true);
      rewriter.eraseOp(trans_op);
      return success();
    }
    return failure();
  }
};

Value is_reshape_permute(Value in) {
  auto reshape0 = dyn_cast<top::ReshapeOp>(in.getDefiningOp());
  if (!reshape0 || !reshape0->hasOneUse()) {
    return NULL;
  }
  auto permute0 = dyn_cast<top::PermuteOp>(reshape0.getInput().getDefiningOp());
  if (!permute0 || !permute0->hasOneUse()) {
    return NULL;
  }
  auto reshape1 = dyn_cast<top::ReshapeOp>(permute0.getInput().getDefiningOp());
  if (!reshape1) {
    return permute0.getInput();
  } else if (!reshape1->hasOneUse()) {
    return NULL;
  } else {
    return reshape1.getInput();
  }
}

Value is_permute_reshape(Value in) {
  Value permute_out;
  auto reshape0 = dyn_cast<top::ReshapeOp>(in.getDefiningOp());
  if (!reshape0) {
    permute_out = in;
  } else if (!reshape0->hasOneUse()) {
    return NULL;
  } else {
    permute_out = reshape0.getInput();
  }
  auto permute0 = dyn_cast<top::PermuteOp>(permute_out.getDefiningOp());
  if (!permute0 || !permute0->hasOneUse())
    return NULL;
  auto reshape1 = dyn_cast<top::ReshapeOp>(permute0.getInput().getDefiningOp());
  if (!reshape1 || !reshape1->hasOneUse()) {
    return NULL;
  }
  return reshape1.getInput();
}

class ConvertMatMul2Attention : public OpRewritePattern<top::MatMulOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(top::MatMulOp op,
                                PatternRewriter &rewriter) const override {
    auto filter = op.getRight();
    // return failure();
    if (module::isWeight(filter) == false) {
      return failure();
    }
    if (op->hasOneUse() == false) {
      return failure();
    }
    Value matmul_out = is_reshape_permute(op.getInput());
    if (matmul_out == NULL) {
      return failure();
    }
    auto matmul1 = dyn_cast<top::MatMulOp>(matmul_out.getDefiningOp());
    if (!matmul1) {
      return failure();
    }
    auto softmax = dyn_cast<top::SoftmaxOp>(matmul1.getInput().getDefiningOp());
    if (!softmax || !softmax->hasOneUse()) {
      return failure();
    }
    Value mul_out;
    auto add = dyn_cast<top::AddOp>(softmax.getInput().getDefiningOp());
    if (!add) {
      mul_out = softmax.getInput();
    } else {
      mul_out = add.getInputs()[0];
    }
    auto mul_const = dyn_cast<top::MulConstOp>(mul_out.getDefiningOp());
    if (!mul_const || !mul_const->hasOneUse()) {
      return failure();
    }
    auto matmul0 = dyn_cast<top::MatMulOp>(mul_const.getInput().getDefiningOp());
    if (!matmul0) {
      return failure();
    }
    // queries
    Value matmul_out1 = is_permute_reshape(matmul0.getInput());
    if (matmul_out1 == NULL) {
      return failure();
    }
    auto matmul_queries = dyn_cast<top::MatMulOp>(matmul_out1.getDefiningOp());
    if (!matmul_queries || !module::isWeight(matmul_queries.getRight())) {
      return failure();
    }
    // keys
    auto permute0 = dyn_cast<top::PermuteOp>(matmul0.getRight().getDefiningOp());
    if (!permute0 || !permute0->hasOneUse())
      return failure();
    Value matmul_out2 = is_permute_reshape(permute0.getInput());
    if (matmul_out2 == NULL) {
      return failure();
    }
    auto matmul_keys = dyn_cast<top::MatMulOp>(matmul_out2.getDefiningOp());
    if (!matmul_keys || !module::isWeight(matmul_keys.getRight())) {
      return failure();
    }
    // values
    Value matmul_out3 = is_permute_reshape(matmul1.getRight());
    if (matmul_out3 == NULL) {
      return failure();
    }
    auto matmul_values = dyn_cast<top::MatMulOp>(matmul_out3.getDefiningOp());
    if (!matmul_values || !module::isWeight(matmul_values.getRight())) {
      return failure();
    }
    if (module::isBM1686()) {
      auto len = module::getNumElements(matmul_queries.getInput());
      // TODO: do not suppose attention when size greater than [batch, 2048, 320]
      if (len > 2048 * 320) {
        return failure();
      }
    }
    rewriter.setInsertionPointAfter(op);
    auto none = module::getNoneOp(op);
    std::vector<NamedAttribute> attrs;
    attrs.push_back(rewriter.getNamedAttr("scale", mul_const.getConstValAttr()));
    auto batch = module::getShape(op.getOutput())[0];
    auto shape = module::getShape(matmul1.getOutput());
    int64_t head;
    if (shape.size() == 3) {
      head = shape[0] / batch;
    } else {
      head = shape[1];
    }
    attrs.push_back(rewriter.getNamedAttr("head", rewriter.getI64IntegerAttr(head)));
    if (module::isCalibratedType(op.getOutput().getType())) {
      // quant param
      // qo, ko, vo, m0, si, so, m1
      std::vector<double> scale_v;
      double scale;
      int64_t zp;
      module::getScaleAndZeroPoint(matmul_queries.getOutput(), scale, zp, false);
      scale_v.push_back(scale);
      module::getScaleAndZeroPoint(matmul_keys.getOutput(), scale, zp, false);
      scale_v.push_back(scale);
      module::getScaleAndZeroPoint(matmul_values.getOutput(), scale, zp, false);
      scale_v.push_back(scale);
      module::getScaleAndZeroPoint(matmul0.getOutput(), scale, zp, false);
      scale_v.push_back(scale);
      module::getScaleAndZeroPoint(mul_const.getOutput(), scale, zp, false);
      scale_v.push_back(scale);
      module::getScaleAndZeroPoint(softmax.getOutput(), scale, zp, false);
      scale_v.push_back(scale);
      module::getScaleAndZeroPoint(matmul1.getOutput(), scale, zp, false);
      scale_v.push_back(scale);
      attrs.push_back(rewriter.getNamedAttr("scale_param", rewriter.getF64ArrayAttr(scale_v)));
    }
    std::vector<Value> operands;
    operands.push_back(matmul_queries.getInput());
    operands.push_back(matmul_keys.getInput());
    operands.push_back(matmul_values.getInput());
    operands.push_back(matmul_queries.getRight());
    operands.push_back(matmul_queries.getBias());
    operands.push_back(matmul_keys.getRight());
    operands.push_back(matmul_keys.getBias());
    operands.push_back(matmul_values.getRight());
    operands.push_back(matmul_values.getBias());
    operands.push_back(op.getRight());
    operands.push_back(op.getBias());
    operands.push_back(add ? add.getInputs()[1] : none);
    auto attention =
        rewriter.create<top::AttentionOp>(op.getLoc(), op.getOutput().getType(),
                                       operands, attrs);
    op.replaceAllUsesWith(attention.getOperation());
    rewriter.eraseOp(op);
    return success();
  }
};

// reorder op when reshapeOp is before matmul/mulconst/cast/softmax op to
// eliminate reshapeOp
class ReshapeReorderPattern : public OpRewritePattern<top::ReshapeOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(top::ReshapeOp op,
                                PatternRewriter &rewriter) const override {
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
      auto new_loc =
          NameLoc::get(rewriter.getStringAttr(ori_name + "_Reshape"));
      next_op->setLoc(new_loc);

      // Add ReshapeOp after MatMul
      rewriter.setInsertionPointAfterValue(next_out);
      auto ori_loc = NameLoc::get(rewriter.getStringAttr(ori_name));
      auto new_reshape_op = rewriter.create<top::ReshapeOp>(
          ori_loc, ori_out_type, ValueRange{next_out});
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
      auto new_loc =
          NameLoc::get(rewriter.getStringAttr(ori_name + "_Reshape"));
      next_op_->setLoc(new_loc);

      // Add ReshapeOp after MulConst/Cast/Softmax
      rewriter.setInsertionPointAfterValue(next_out);
      auto ori_loc = NameLoc::get(rewriter.getStringAttr(ori_name));
      auto new_reshape_op = rewriter.create<top::ReshapeOp>(
          ori_loc, ori_out_type, ValueRange{next_out});
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
};

} // namespace bm1684x

namespace top {
using namespace bm1684x;
void populateOptimizeBM1684XPatterns(RewritePatternSet *patterns) {
  // clang-format off
  patterns->add<
      ConvertMatMulWithRightTranspose,
      ConvertMatMul2Attention,
      ReshapeReorderPattern
  >(patterns->getContext());
  // clang-format on
}
} // namespace top
} // namespace tpu_mlir
