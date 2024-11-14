//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//
#include "tpu_mlir/Dialect/Tpu/Transforms/DevParallel/Distribute.h"
#include "tpu_mlir/Dialect/Tpu/Transforms/DevParallel/DistributeUtils.h"

// ======================================
// pattern MatMulSliceMerge3:
// Attention is added with FFN
// e.g. Falcon
// ======================================

namespace tpu_mlir {
namespace tpu {

// in -> Acitve -> out
// in -> Cast -> Active -> Cast -> out
static Value isCastActive(Value in) {
  auto active_op = in.getDefiningOp();
  if (dyn_cast<tpu::ActiveOp>(active_op)) {
    return active_op->getOperand(0);
  } else if (auto cast_out_op = dyn_cast<tpu::CastOp>(active_op)) {
    active_op = dyn_cast<tpu::ActiveOp>(cast_out_op.getInput().getDefiningOp());
    if (!active_op) {
      return nullptr;
    }
    auto cast_in_op =
        dyn_cast<tpu::CastOp>(active_op->getOperand(0).getDefiningOp());
    if (cast_in_op) {
      return cast_in_op.getInput();
    }
  }
  return nullptr;
}

static Operation *isFFNPattern(Operation *op) {
  if (!isLargeMatMul(op) || !op->hasOneUse()) {
    return nullptr;
  }
  // ActiveOp
  Value active_value = isCastActive(op->getOperand(0));
  if (!active_value) {
    return nullptr;
  }
  // MatMulOp
  auto matmul_op = dyn_cast<tpu::MatMulOp>(active_value.getDefiningOp());
  if (!matmul_op) {
    return nullptr;
  }

  // LayerNormOp or CastOp
  auto top_op = matmul_op->getOperand(0).getDefiningOp();
  if (isa<tpu::CastOp>(top_op)) {
    top_op = top_op->getOperand(0).getDefiningOp();
  }
  if (isa<tpu::LayerNormOp>(top_op)) {
    top_op = dyn_cast<tpu::LayerNormOp>(top_op);
  } else {
    return nullptr;
  }

  return top_op;
}

LogicalResult
MatMulSliceMerge3::matchAndRewriteImpl(tpu::AddOp op,
                                       PatternRewriter &rewriter) const {
  if (!op->hasOneUse() || module::isOpInDevParallel(op)) {
    return failure();
  }

  Operation *end_op = op;
  auto user = *op->user_begin();
  if (isa<tpu::CastOp>(user)) {
    end_op = user;
  }

  auto in0 = op.getInputs()[0];
  Operation *begin_op = op;
  if (isa<tpu::CastOp>(in0.getDefiningOp())) {
    begin_op = in0.getDefiningOp();
    in0 = begin_op->getOperand(0);
  }
  if (!isa<top::InputOp>(in0.getDefiningOp())) {
    return failure();
  }

  auto op1 = op.getInputs()[1].getDefiningOp();
  Operation *ffn_op;
  Operation *attn_op;
  if (isa<tpu::AddOp>(op1)) {
    ffn_op = op1->getOperand(0).getDefiningOp();
    attn_op = op1->getOperand(1).getDefiningOp();
  } else if (isa<tpu::MatMulOp>(op1)) {
    ffn_op = op1->getOperand(2).getDefiningOp();
    attn_op = op1;
  }

  if (!isa<tpu::MatMulOp>(ffn_op) || !isa<tpu::MatMulOp>(attn_op)) {
    return failure();
  }

  std::vector<Operation *> attn_begin_ops, attn_end_ops;
  auto ffn_begin_op = isFFNPattern(ffn_op);
  if (ffn_begin_op == nullptr) {
    return failure();
  }
  int num_head = 0;
  if (!isAttentionPattern(attn_op, attn_begin_ops, attn_end_ops, true,
                          &num_head)) {
    return failure();
  }

  if (ffn_begin_op->getOperand(0) != attn_begin_ops[0]->getOperand(0)) {
    return failure();
  }

  // Bingo !!
  std::vector<Operation *> begin_ops{begin_op};
  begin_ops.insert(begin_ops.end(), attn_begin_ops.begin() + 1,
                   attn_begin_ops.end());
  std::vector<int64_t> begin_methods;
  begin_methods.push_back(1);
  begin_methods.push_back(1);
  begin_methods.push_back(1);
  begin_methods.push_back(1);
  if (begin_ops.size() > 4) {
    begin_methods.push_back(2);
    begin_methods.push_back(2);
  }

  std::vector<Operation *> end_ops = {end_op};
  end_ops.insert(end_ops.end(), std::next(attn_end_ops.begin()),
                 attn_end_ops.end());
  std::vector<int64_t> end_methods;
  end_methods.push_back(1);
  end_methods.push_back(3);
  end_methods.push_back(3);
  distribute(rewriter, begin_ops, end_ops, tpu::DevPattern::MatMulSliceMerge3,
             begin_methods, end_methods, num_head);

  return success();
}

void sliceMerge3Split(PatternRewriter &rewriter, tpu::DevBeginOp op,
                      int64_t num_devices) {
  // users: hidden_state, pos_ids_gather0, pos_ids_gather1, attn_mask,
  // past_k, past_v
  std::vector<Operation *> users(op->user_begin(), op->user_end());
  auto residual = users[0];
  std::vector<Operation *> use_ops(residual->user_begin(),
                                   residual->user_end());
  auto ffn_ln = use_ops[1];
  auto attn_ln = use_ops[2];
  bool decode_phase = (users.size() > 5);
  int num_head = op.getNumHead();

  std::vector<Value> end_operands;
  std::vector<Value> operands;
  Operation *end_op = nullptr;
  Operation *next_op;
  Value cur_out;
  std::vector<Value> other_opds;
  for (int cur_device = 0; cur_device < num_devices; ++cur_device) {
    auto suffix = std::to_string(cur_device);
    // clone residual branch
    next_op = residual;
    cur_out = next_op->getOperand(0);
    if (isa<tpu::CastOp>(next_op)) {
      next_op = cloneCommonOp(rewriter, next_op, cur_out, suffix);
    }
    auto ln_input = cur_out;
    createMulConstOp(rewriter, cur_out, cur_device, cur_device == 0 ? 1.0 : 0);
    auto residual_out = cur_out;

    // clone pos_ids input branch
    std::vector<Value> pos_ids;
    next_op = users[1];
    cur_out = next_op->getOperand(1);
    auto muls0 = cloneOpWithWeight(rewriter, next_op, cur_out, suffix);
    pos_ids.push_back(cur_out);

    next_op = users[2];
    cur_out = next_op->getOperand(1);
    auto muls1 = cloneOpWithWeight(rewriter, next_op, cur_out, suffix);
    pos_ids.push_back(cur_out);
    if (!isa<tpu::ConcatOp>(muls0[0]->getOperand(0).getDefiningOp())) {
      std::swap(pos_ids[0], pos_ids[1]);
    }

    // clone attn_mask input branch
    next_op = users[3];
    cur_out = next_op->getOperand(0);
    // (Cast) -> Reshape
    while (isa<tpu::CastOp, tpu::ReshapeOp>(next_op)) {
      next_op = cloneCommonOp(rewriter, next_op, cur_out, -1, num_devices,
                              cur_device)[0];
    }
    auto attn_mask_out = cur_out;

    Value past_k_out, past_v_out;
    if (decode_phase) {
      // clone past_k, past_v
      for (int j = users.size() - 2; j < users.size(); j++) {
        next_op = users[j];
        cur_out = next_op->getOperand(0);
        next_op = createSliceOp(rewriter, next_op, cur_out, 2, num_devices,
                                cur_device, num_head);
        if (isa<tpu::CastOp>(next_op)) {
          next_op = cloneCommonOp(rewriter, next_op, cur_out, 2, num_devices,
                                  cur_device)[0];
        }
        if (j == users.size() - 2) {
          past_k_out = cur_out;
        } else {
          past_v_out = cur_out;
        }
      }
    }

    // FFN
    next_op = ffn_ln;
    cur_out = ln_input;
    next_op = cloneOpWithWeight(rewriter, next_op, cur_out, suffix)[0];
    next_op = cloneColParallelMatMul(rewriter, next_op, cur_out, num_devices,
                                     cur_device, num_head);
    next_op = cloneCommonOp(rewriter, next_op, cur_out, suffix);
    next_op = cloneRowParallelMatMul(rewriter, next_op, cur_out, num_devices,
                                     cur_device, num_head);
    Value ffn_out = cur_out;

    // Attention
    // Attention Input branch
    next_op = attn_ln;
    cur_out = ln_input;
    std::vector<Operation *> reshape_users = cloneAttentionInput(
        rewriter, next_op, cur_out, num_devices, cur_device, num_head);
    Value start_out = cur_out;
    // Value branch
    std::vector<Value> outs;
    outs.clear();
    next_op = reshape_users[0];
    other_opds = {past_v_out};
    next_op =
        cloneAttentionValue(rewriter, next_op, cur_out, other_opds, outs, 2,
                            num_devices, cur_device, true, num_head)[0];
    Value value_out = outs[0];
    Value value_branch = outs[1];

    // Query branch
    next_op = reshape_users[2];
    cur_out = start_out;
    next_op = cloneAttentionQuery(rewriter, next_op, cur_out, pos_ids,
                                  num_devices, cur_device, true, num_head)[0];
    Value query_branch = cur_out;
    auto query_next_op = next_op;
    // Key branch
    outs.clear();
    cur_out = start_out;
    next_op = reshape_users[1];
    other_opds = {pos_ids[0], pos_ids[1], past_k_out};
    next_op = cloneAttentionKey(rewriter, next_op, cur_out, other_opds, outs,
                                num_devices, cur_device, true, num_head)[0];
    Value key_out = outs[0];
    Value key_branch = outs[1];
    assert(query_next_op == next_op);
    // Q@K
    operands.clear();
    operands.push_back(query_branch);
    operands.push_back(key_branch);
    next_op = cloneMultiInsOp(rewriter, next_op, cur_out, operands, 2,
                              num_devices, cur_device);
    // Attention Matrix branch
    other_opds = {attn_mask_out};
    next_op = cloneAttentionMatrix(rewriter, next_op, cur_out, 2, other_opds,
                                   num_devices, cur_device)[0];
    Value qk_out = cur_out;
    // QK@V
    operands.clear();
    operands.push_back(qk_out);
    operands.push_back(value_branch);
    next_op = cloneMultiInsOp(rewriter, next_op, cur_out, operands, 2,
                              num_devices, cur_device);
    // Attention Output branch
    next_op = cloneAttentionOutput(rewriter, next_op, cur_out, num_devices,
                                   cur_device, num_head)[0];
    Value attn_out = cur_out;

    // out0 = ffn_out + attn_out
    auto ffn_out_shape = module::getShape(ffn_out);
    if (ffn_out_shape.size() == 3 && ffn_out_shape[0] == 1 &&
        ffn_out_shape[1] == 1) {
      auto matmul = attn_out.getDefiningOp();
      matmul->setOperand(2, ffn_out);
    } else {
      operands.clear();
      operands.push_back(ffn_out);
      operands.push_back(attn_out);
      next_op = cloneMultiInsOp(rewriter, next_op, cur_out, operands, -1,
                                num_devices, cur_device);
    }
    // out0 + residual_out
    operands.clear();
    operands.push_back(cur_out);
    operands.push_back(residual_out);
    next_op = cloneMultiInsOp(rewriter, next_op, cur_out, operands, -1,
                              num_devices, cur_device);
    if (isa<tpu::CastOp>(next_op)) {
      next_op = cloneCommonOp(rewriter, next_op, cur_out, suffix);
    }

    end_operands.push_back(cur_out);
    end_operands.push_back(key_out);
    end_operands.push_back(value_out);
    if (cur_device == 0) {
      end_op = next_op;
    } else {
      assert(end_op == next_op);
    }
  }

  assert(isa<tpu::DevEndOp>(end_op));
  end_op->setOperands(end_operands);

  for (size_t i = 0; i < users.size(); ++i) {
    eraseForward(rewriter, users[i]);
  }
  module::removeUnusedOp();
}

} // namespace tpu
} // namespace tpu_mlir
