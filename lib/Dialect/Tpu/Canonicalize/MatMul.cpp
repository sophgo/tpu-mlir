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
#include "tpu_mlir/Support/OpRewriterPatternEx.h"

namespace tpu_mlir {
namespace tpu {
// MatMul(fp16) + Add(fp16) => MatMul
struct MatMulWithBias : public OpRewriterPatternEx<tpu::MatMulOp> {
  using OpRewriterPatternEx::OpRewriterPatternEx;

  MatMulWithBias(mlir::MLIRContext *context)
      : OpRewriterPatternEx<tpu::MatMulOp>(context, "MatMulWithBias") {}

  LogicalResult matchAndRewriteImpl(tpu::MatMulOp op,
                                    PatternRewriter &rewriter) const override {
    auto out = op.getOutput();
    if (!out.hasOneUse() || !module::isNone(op.getBias())) {
      return failure();
    }
    auto out_stype = module::getStorageType(out);
    if (!out_stype.isa<FloatType>()) {
      return failure();
    }
    if ((module::isBM1688() || module::isMARS3() || module::isSGTPUV8() ||
         module::isSG2380()) &&
        !out_stype.isF32()) {
      // only f32 support
      return failure();
    }
    auto user = *out.user_begin();
    auto add_op = dyn_cast_or_null<tpu::AddOp>(user);
    if (!add_op || add_op.getNumOperands() != 2) {
      return failure();
    }
    auto another = add_op.getInputs()[0];
    if (another == out) {
      another = add_op.getInputs()[1];
    }
    auto o_shape = module::getShape(out);
    auto a_shape = module::getShape(another);
    auto num_elem = module::getNumElements(another);
    if (a_shape.back() != num_elem || o_shape.back() != num_elem ||
        o_shape.size() == 1) {
      return failure();
    }
    op->setOperand(2, another);
    op->setLoc(add_op.getLoc());
    op->moveAfter(add_op);
    add_op->replaceAllUsesWith(op);
    rewriter.eraseOp(add_op);
    return success();
  }
};

struct MatMul2FAttention : public OpRewriterPatternEx<tpu::MatMulOp> {
  using OpRewriterPatternEx::OpRewriterPatternEx;
  MatMul2FAttention(mlir::MLIRContext *context)
      : OpRewriterPatternEx<tpu::MatMulOp>(context, "MatMul2FAttention") {}
  LogicalResult matchAndRewriteImpl(tpu::MatMulOp op,
                                    PatternRewriter &rewriter) const override {
    // return failure();
    std::vector<Operation *> op_need_del;
    if (!module::isBM1684X() && !module::isBM1688() &&
        !module::isBM1690Family()) {
      return failure();
    }

    auto out_type = module::getStorageType(op.getOutput());
    if (!out_type.isBF16() && !out_type.isF16()) {
      return failure();
    }
    if (op->hasOneUse() == false) {
      return failure();
    }

    // forward
    tpu::ReshapeOp reshape_op;
    auto o_permute =
        dyn_cast<tpu::PermuteOp>(*(op.getOutput().getUsers().begin()));
    // (*(op.getOutput().getUsers().begin()))->dump();
    if (!o_permute) {
      return failure();
    } else {
      if (!o_permute->hasOneUse()) {
        return failure();
      }
      auto o_permute_order = module::getI64Array(o_permute.getOrder());
      if (o_permute_order->size() != 4 || o_permute_order->at(0) != 0 ||
          o_permute_order->at(1) != 2 || o_permute_order->at(2) != 1 ||
          o_permute_order->at(3) != 3) {
        return failure();
      }
      reshape_op =
          dyn_cast<tpu::ReshapeOp>(*(o_permute.getOutput().getUsers().begin()));
    }
    if (!reshape_op || !reshape_op->hasOneUse()) {
      return failure();
    }
    op_need_del.emplace_back(reshape_op);
    if (o_permute) {
      op_need_del.emplace_back(o_permute);
    }
    op_need_del.emplace_back(op);

    // backward
    tpu::SoftmaxOp softmax;
    if (auto cast_op = dyn_cast<tpu::CastOp>(op.getInput().getDefiningOp())) {
      if (!cast_op->hasOneUse()) {
        return failure();
      }
      softmax = dyn_cast<tpu::SoftmaxOp>(cast_op.getInput().getDefiningOp());
      op_need_del.emplace_back(cast_op);
    } else {
      softmax = dyn_cast<tpu::SoftmaxOp>(op.getInput().getDefiningOp());
    }
    if (!softmax || !softmax->hasOneUse()) {
      return failure();
    }
    op_need_del.emplace_back(softmax);
    Value mul_out;
    tpu::AddOp add;
    tpu::CastOp cast_op =
        dyn_cast<tpu::CastOp>(softmax.getInput().getDefiningOp());
    if (cast_op) {
      if (!cast_op->hasOneUse()) {
        return failure();
      }
      add = dyn_cast<tpu::AddOp>(cast_op.getInput().getDefiningOp());
      op_need_del.emplace_back(cast_op);
    } else {
      add = dyn_cast<tpu::AddOp>(softmax.getInput().getDefiningOp());
    }
    if (!add) {
      if (cast_op) {
        mul_out = cast_op.getInput();
      } else {
        mul_out = softmax.getInput();
      }
    } else {
      mul_out = add.getInputs()[0];
      op_need_del.emplace_back(add);
    }
    auto mul_const = dyn_cast<tpu::MulConstOp>(mul_out.getDefiningOp());
    if (!mul_const || !mul_const->hasOneUse()) {
      return failure();
    }
    op_need_del.emplace_back(mul_const);
    auto matmul0 =
        dyn_cast<tpu::MatMulOp>(mul_const.getInput().getDefiningOp());
    if (!matmul0) {
      return failure();
    }
    op_need_del.emplace_back(matmul0);
    // queries
    Value q_in;
    auto q_permute =
        dyn_cast<tpu::PermuteOp>(matmul0.getInput().getDefiningOp());
    if (!q_permute || !q_permute->hasOneUse()) {
      return failure();
    }
    op_need_del.emplace_back(q_permute);
    q_in = q_permute.getInput();

    // keys
    auto k_permute =
        dyn_cast<tpu::PermuteOp>(matmul0.getRight().getDefiningOp());
    auto k_reshape =
        dyn_cast<tpu::ReshapeOp>(matmul0.getRight().getDefiningOp());
    bool has_tile = false;
    Value k_in;
    std::shared_ptr<std::vector<int64_t>> k_permute_order;

    if (k_permute) {
      if (!k_permute->hasOneUse()) {
        return failure();
      }
      k_permute_order = module::getI64Array(k_permute.getOrder());
      op_need_del.emplace_back(k_permute);
      k_in = k_permute.getInput();
      /// keys tile
      auto k_reshape = dyn_cast<tpu::ReshapeOp>(k_in.getDefiningOp());
      if (k_reshape) {
        auto k_tile = k_reshape.getInput().getDefiningOp();
        if (isa<tpu::MulOp, tpu::TileOp>(k_tile)) {
          auto k_unsqu =
              dyn_cast<tpu::UnsqueezeOp>(k_tile->getOperand(0).getDefiningOp());
          if (k_unsqu) {
            has_tile = true;
            op_need_del.emplace_back(k_reshape);
            op_need_del.emplace_back(k_tile);
            op_need_del.emplace_back(k_unsqu);
            k_in = k_unsqu.getInput();
          } else {
            return failure();
          }
        }
      }
    } else if (k_reshape) {
      if (!k_reshape->hasOneUse()) {
        return failure();
      }
      op_need_del.emplace_back(k_reshape);
      k_in = k_reshape.getInput();
      /// keys tile
      auto k_tile = dyn_cast<tpu::TileOp>(k_reshape.getInput().getDefiningOp());
      if (k_tile && k_tile->hasOneUse()) {
        auto k_unsqu =
            dyn_cast<tpu::UnsqueezeOp>(k_tile->getOperand(0).getDefiningOp());
        if (k_unsqu && k_unsqu->hasOneUse()) {
          auto k_permute =
              dyn_cast<tpu::PermuteOp>(k_unsqu.getInput().getDefiningOp());
          if (k_permute && k_permute->hasOneUse()) {
            has_tile = true;
            op_need_del.emplace_back(k_tile);
            op_need_del.emplace_back(k_unsqu);
            op_need_del.emplace_back(k_permute);
            k_in = k_permute.getInput();
            k_permute_order = module::getI64Array(k_permute.getOrder());
          } else {
            return failure();
          }
        } else {
          return failure();
        }
      } else {
        return failure();
      }
    } else {
      return failure();
    }

    // Avoid getting into wrong FAttention
    auto right_trans = matmul0.getRightTranspose();
    if (right_trans) {
      if (k_permute_order->size() != 4 || k_permute_order->at(0) != 0 ||
          k_permute_order->at(1) != 2 || k_permute_order->at(2) != 1 ||
          k_permute_order->at(3) != 3) {
        return failure();
      }
    } else {
      if (k_permute_order->size() != 4 || k_permute_order->at(0) != 0 ||
          k_permute_order->at(1) != 2 || k_permute_order->at(2) != 3 ||
          k_permute_order->at(3) != 1) {
        return failure();
      }
    }
    auto q_permute_order = module::getI64Array(q_permute.getOrder());
    if (q_permute_order->size() != 4 || q_permute_order->at(0) != 0 ||
        q_permute_order->at(1) != 2 || q_permute_order->at(2) != 1 ||
        q_permute_order->at(3) != 3) {
      return failure();
    }

    // values
    auto v_permute = dyn_cast<tpu::PermuteOp>(op.getRight().getDefiningOp());
    auto v_reshape = dyn_cast<tpu::ReshapeOp>(op.getRight().getDefiningOp());
    Value v_in;

    if (v_permute) {
      if (!v_permute->hasOneUse()) {
        return failure();
      }
      op_need_del.emplace_back(v_permute);
      v_in = v_permute.getInput();
      /// values tile
      auto v_reshape = dyn_cast<tpu::ReshapeOp>(v_in.getDefiningOp());
      if (v_reshape) {
        auto v_tile = v_reshape.getInput().getDefiningOp();
        if (isa<tpu::MulOp, tpu::TileOp>(v_tile)) {
          auto v_unsqu =
              dyn_cast<tpu::UnsqueezeOp>(v_tile->getOperand(0).getDefiningOp());
          if (v_unsqu) {
            if (!has_tile) {
              return failure();
            }
            op_need_del.emplace_back(v_reshape);
            op_need_del.emplace_back(v_tile);
            op_need_del.emplace_back(v_unsqu);
            v_in = v_unsqu.getInput();
          } else {
            return failure();
          }
        }
      }
    } else if (v_reshape) {
      if (!v_reshape->hasOneUse()) {
        return failure();
      }
      op_need_del.emplace_back(v_reshape);
      auto v_tile = dyn_cast<tpu::TileOp>(v_reshape.getInput().getDefiningOp());
      if (v_tile && v_tile->hasOneUse()) {
        auto v_unsqu =
            dyn_cast<tpu::UnsqueezeOp>(v_tile->getOperand(0).getDefiningOp());
        if (v_unsqu && v_unsqu->hasOneUse()) {
          if (!has_tile) {
            return failure();
          }
          auto v_permute =
              dyn_cast<tpu::PermuteOp>(v_unsqu.getInput().getDefiningOp());
          if (v_permute && v_permute->hasOneUse()) {
            op_need_del.emplace_back(v_tile);
            op_need_del.emplace_back(v_unsqu);
            op_need_del.emplace_back(v_permute);
            v_in = v_permute.getInput();
          } else {
            return failure();
          }
        } else {
          return failure();
        }
      } else {
        return failure();
      }
    } else {
      return failure();
    }

    if (module::getShape(k_in) != module::getShape(v_in)) {
      return failure();
    }
    rewriter.setInsertionPointAfter(reshape_op);
    auto o_shape = module::getShape(op.getOutput());
    auto sf_shape = module::getShape(softmax.getInput());
    auto kv_shape = module::getShape(k_in);
    auto none = module::getNoneOp(op);
    int64_t q_head, kv_head;
    int64_t d;
    int64_t mq;
    int64_t mk;
    int64_t batch;

    assert(o_shape.size() == 4 && sf_shape.size() == 4);
    batch = o_shape[0];
    q_head = o_shape[1];
    kv_head = kv_shape[2];
    d = o_shape[3];
    mq = sf_shape[2];
    mk = sf_shape[3];
    assert(o_shape[2] == mq && sf_shape[1] == q_head);

    // ppl flash attention only support d <= 256, bf16 & fp16
    if (d > 128 || mk < 4) {
      return failure();
    }
    if ((module::isBM1684X() && (q_head / kv_head > 16)) ||
        (module::isBM1688() && (q_head / kv_head > 8)) ||
        (module::isBM1690Family() && (q_head / kv_head > 16))) {
      return failure();
    }
    if (add) {
      auto add_shape = module::getShape(add.getInputs()[1]);
      if (add_shape[0] != batch || add_shape[2] != mq || add_shape[3] != mk) {
        return failure();
      }
    }
    std::vector<NamedAttribute> attrs;
    attrs.push_back(
        rewriter.getNamedAttr("scale", mul_const.getConstValAttr()));
    attrs.push_back(
        rewriter.getNamedAttr("q_head", rewriter.getI64IntegerAttr(q_head)));
    attrs.push_back(
        rewriter.getNamedAttr("kv_head", rewriter.getI64IntegerAttr(kv_head)));
    attrs.push_back(
        rewriter.getNamedAttr("dim", rewriter.getI64IntegerAttr(d)));
    attrs.push_back(
        rewriter.getNamedAttr("batch", rewriter.getI64IntegerAttr(batch)));
    attrs.push_back(
        rewriter.getNamedAttr("mq", rewriter.getI64IntegerAttr(mq)));
    attrs.push_back(
        rewriter.getNamedAttr("mk", rewriter.getI64IntegerAttr(mk)));
    std::vector<Value> operands;
    operands.push_back(q_in);

    operands.push_back(k_in);
    operands.push_back(v_in);
    operands.push_back(add ? add.getInputs()[1] : none);
    operands.push_back(none);
    auto attention = rewriter.create<tpu::FAttentionOp>(
        reshape_op.getLoc(), reshape_op.getOutput().getType(), operands, attrs);
    reshape_op.replaceAllUsesWith(attention.getOperation());
    for (auto op : op_need_del) {
      rewriter.eraseOp(op);
    }
    return success();
  }
};

struct MatMulFuseRequant : public OpRewriterPatternEx<tpu::MatMulOp> {
  using OpRewriterPatternEx::OpRewriterPatternEx;

  MatMulFuseRequant(mlir::MLIRContext *context)
      : OpRewriterPatternEx<tpu::MatMulOp>(context, "MatMulFuseRequant") {}

  LogicalResult matchAndRewriteImpl(tpu::MatMulOp op,
                                    PatternRewriter &rewriter) const override {
    auto out = op.getOutput();
    if (!out.hasOneUse()) {
      return failure();
    }
    auto nextOp = *op.getOutput().user_begin();
    if (!isa<tpu::RequantIntOp>(nextOp))
      return failure();

    auto requantOp = dyn_cast<tpu::RequantIntOp>(nextOp);
    auto quant_mode = requantOp.getQuantMode();
    if (quant_mode == tpu::RequantMode::TFLite_LShift ||
        quant_mode == tpu::RequantMode::TFLite)
      return failure();

    rewriter.setInsertionPoint(op);
    std::vector<mlir::Value> operands(op.getOperands().begin(),
                                      op.getOperands().end());
    auto newOp = rewriter.create<tpu::MatMulOp>(requantOp->getLoc(),
                                                requantOp.getOutput().getType(),
                                                operands, op->getAttrs());

    newOp.setMultipliersAttr(
        rewriter.getI64ArrayAttr(requantOp.getMultiplier()));
    newOp.setRshiftsAttr(rewriter.getI64ArrayAttr(requantOp.getRshift()));
    newOp.setQuantModeAttr(requantOp.getQuantModeAttr());

    auto round_mode = requantOp.getRoundModeAttr().getValue();
    newOp.setRoundMode(round_mode);
    rewriter.replaceOp(requantOp, newOp.getResult());
    rewriter.eraseOp(op);
    return success();
  }
};

void tpu::MatMulOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                                MLIRContext *context) {
  results.insert<MatMulWithBias, MatMul2FAttention, MatMulFuseRequant>(context);
}

} // namespace tpu
} // namespace tpu_mlir
