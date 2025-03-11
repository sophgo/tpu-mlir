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
// pattern MatMulSliceMerge2
// e.g. Llama2
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

template <typename MatMulTy>
LogicalResult MatMulSliceMerge2<MatMulTy>::matchAndRewriteImpl(
    MatMulTy op, PatternRewriter &rewriter) const {
  if (!isLargeMatMul(op) || module::isOpInDevParallel(op)) {
    return failure();
  }
  if (op->hasOneUse() == false) {
    return failure();
  }

  Operation *residual_begin_op = op;
  Operation *end_op = op;
  if (isa<tpu::AddOp>(*end_op->user_begin())) {
    end_op = *end_op->user_begin();
    residual_begin_op = end_op;
  }
  if (isa<tpu::CastOp>(*end_op->user_begin())) {
    end_op = *end_op->user_begin();
  }

  // MulOp
  auto mul_down_op = dyn_cast<tpu::MulOp>(op.getOperand(0).getDefiningOp());
  if (!mul_down_op) {
    return failure();
  }
  // ActiveOp
  int matmul_idx = 1;
  auto in = mul_down_op.getOperand(0);
  if (isa<MatMulOp, A16MatMulOp>(in.getDefiningOp())) {
    in = mul_down_op.getOperand(1);
    matmul_idx = 0;
  }
  Value active_value = isCastActive(in);
  if (!active_value) {
    return failure();
  }
  // MatMulOp / A16MatMulOp
  auto mul_right_op = mul_down_op.getOperand(matmul_idx).getDefiningOp();
  auto right_mm_op = dyn_cast<tpu::MatMulOp>(mul_right_op);
  auto right_a16mm_op = dyn_cast<tpu::A16MatMulOp>(mul_right_op);
  if (!right_mm_op && !right_a16mm_op) {
    return failure();
  }
  // MatMulOp
  auto mul_left_op = active_value.getDefiningOp();
  auto left_mm_op = dyn_cast<tpu::MatMulOp>(mul_left_op);
  auto left_a16mm_op = dyn_cast<tpu::A16MatMulOp>(mul_left_op);
  if (!left_mm_op && !left_a16mm_op) {
    return failure();
  }
  // RMSNormOp or CastOp
  auto top_op = left_mm_op ? left_mm_op.getOperand(0).getDefiningOp()
                           : left_a16mm_op.getOperand(0).getDefiningOp();
  if (!isa<tpu::CastOp, tpu::RMSNormOp>(top_op)) {
    return failure();
  }

  // Bingo !!
  std::vector<Operation *> begin_ops{residual_begin_op, top_op};
  std::vector<int64_t> begin_methods{1, 1};
  std::vector<Operation *> end_ops{end_op};
  std::vector<int64_t> end_methods{1};
  distribute(rewriter, begin_ops, end_ops, tpu::DevPattern::MatMulSliceMerge2,
             begin_methods, end_methods, 0);
  return success();
}

template LogicalResult MatMulSliceMerge2<tpu::MatMulOp>::matchAndRewriteImpl(
    tpu::MatMulOp op, PatternRewriter &rewriter) const;

template LogicalResult MatMulSliceMerge2<tpu::A16MatMulOp>::matchAndRewriteImpl(
    tpu::A16MatMulOp op, PatternRewriter &rewriter) const;

void sliceMerge2Split(PatternRewriter &rewriter, tpu::DevBeginOp op,
                      int64_t num_devices) {
  // 1. Define params
  // MatMul params
  std::vector<Operation *> users(op->user_begin(), op->user_end());
  if (isa<tpu::RMSNormOp>(users[0])) {
    std::swap(users[0], users[1]);
  }
  auto residual_op = users[0];
  auto norm_op = users[1];

  std::vector<Value> end_operands;
  std::vector<Value> operands;
  Operation *end_op = nullptr;
  Operation *next_op;
  Value cur_out;
  for (int cur_device = 0; cur_device < num_devices; ++cur_device) {
    auto suffix = std::to_string(cur_device);
    // clone residual branch
    next_op = residual_op;
    cur_out = next_op->getOperand(0);
    if (!isa<tpu::DevBeginOp>(cur_out.getDefiningOp())) {
      if (isa<tpu::AddOp>(next_op)) {
        cur_out = next_op->getOperand(1);
      } else if (isa<tpu::MatMulOp>(next_op)) {
        cur_out = next_op->getOperand(2);
      } else if (isa<tpu::A16MatMulOp>(next_op)) {
        cur_out = next_op->getOperand(3);
      }
    }
    createMulConstOp(rewriter, cur_out, cur_device, cur_device == 0 ? 1.0 : 0);
    auto residual_out = cur_out;

    // MLP
    next_op = norm_op;
    cur_out = next_op->getOperand(0);
    auto norm_users = cloneOpWithWeight(rewriter, next_op, cur_out, suffix);
    // make index=0 is MatMul-> Cast -> Active -> Cast and index=1 is MatMul
    if (isa<tpu::MulOp>(*norm_users[0]->user_begin())) {
      std::swap(norm_users[0], norm_users[1]);
    }
    auto start_out = cur_out;

    next_op = norm_users[0];
    cur_out = start_out;
    next_op = cloneColParallelMatMul(rewriter, next_op, cur_out, num_devices,
                                     cur_device, 0);
    while (isa<tpu::CastOp, tpu::ActiveOp>(next_op)) {
      next_op = cloneCommonOp(rewriter, next_op, cur_out, 2, num_devices,
                              cur_device)[0];
    }
    auto out0 = cur_out;
    auto mul0 = next_op;

    next_op = norm_users[1];
    cur_out = start_out;
    next_op = cloneColParallelMatMul(rewriter, next_op, cur_out, num_devices,
                                     cur_device, 0);
    auto out1 = cur_out;
    assert(mul0 == next_op);

    operands = {out0, out1};
    next_op = cloneMultiInsOp(rewriter, next_op, cur_out, operands, 2,
                              num_devices, cur_device);
    next_op = cloneRowParallelMatMul(rewriter, next_op, cur_out, num_devices,
                                     cur_device, 0);

    if (isa<tpu::AddOp>(next_op)) {
      operands = {residual_out, cur_out};
      next_op = cloneMultiInsOp(rewriter, next_op, cur_out, operands, -1,
                                num_devices, cur_device);
    } else {
      auto mm = cur_out.getDefiningOp();
      int bias_index = isa<tpu::MatMulOp>(mm) ? 2 : 3;
      mm->setOperand(bias_index, residual_out);
    }
    if (isa<tpu::CastOp>(next_op)) {
      next_op = cloneCommonOp(rewriter, next_op, cur_out, -1, num_devices,
                              cur_device)[0];
    }

    end_operands.push_back(cur_out);
    if (cur_device == 0) {
      end_op = next_op;
    } else {
      assert(end_op == next_op);
    }
  }

  // 6. Erase
  assert(isa<tpu::DevEndOp>(end_op));
  end_op->setOperands(end_operands);

  for (size_t i = 0; i < users.size(); ++i) {
    eraseForward(rewriter, users[i]);
  }
  module::removeUnusedOp();
}

/**
 * Attention Tensor Parallelism
 */
template <typename MatMulTy>
LogicalResult AttentionSliceMerge2<MatMulTy>::matchAndRewriteImpl(
    MatMulTy op, PatternRewriter &rewriter) const {
  if (module::isOpInDevParallel(op)) {
    return failure();
  }
  auto add_op = dyn_cast<tpu::AddOp>(*op->user_begin());
  if (!(add_op || !isa<top::NoneOp>(op.getBias().getDefiningOp()))) {
    return failure();
  }

  // residual branch
  Operation *begin_op = add_op;
  Value in1;
  if (add_op) {
    in1 = getTheOtherOperand(add_op, op.getOutput());
  } else {
    begin_op = op;
    in1 = op.getBias();
  }
  if (isa<tpu::CastOp>(in1.getDefiningOp())) {
    begin_op = in1.getDefiningOp();
    in1 = begin_op->getOperand(0);
  }
  if (!isa<top::InputOp>(in1.getDefiningOp())) {
    return failure();
  }

  std::vector<Operation *> attn_begin_ops, attn_end_ops;
  int num_head = 0;
  if (!isAttentionPattern(op, attn_begin_ops, attn_end_ops, true, &num_head)) {
    return failure();
  }

  auto attn_top_op = attn_begin_ops[0];
  bool has_cast = false;
  if (isa<tpu::CastOp>(attn_top_op->getOperand(0).getDefiningOp())) {
    attn_top_op = attn_top_op->getOperand(0).getDefiningOp();
    attn_begin_ops[0] = attn_top_op;
    has_cast = true;
  }

  // residual, attn_ln, pos_id gather0, pos_id gather1, attn_mask op
  std::vector<Operation *> begin_ops;
  if (!has_cast) {
    begin_ops.push_back(begin_op);
  }
  begin_ops.insert(begin_ops.end(), attn_begin_ops.begin(),
                   attn_begin_ops.end());
  std::vector<int64_t> begin_methods{1, 1, 1, 1};
  if (!has_cast) {
    begin_methods.push_back(1);
  }
  if (begin_ops.size() > begin_methods.size()) {
    begin_methods.push_back(2); // past_k op
    begin_methods.push_back(2); // past_v op
  }

  // out_hidden_states op, present_k op; present_v op
  std::vector<Operation *> end_ops = attn_end_ops;
  if (isa<top::NoneOp>(op.getBias().getDefiningOp())) {
    end_ops[0] = add_op;
  }
  std::vector<int64_t> end_methods{1, 3, 3};

  // Bingo !!
  distribute(rewriter, begin_ops, end_ops,
             tpu::DevPattern::AttentionSliceMerge2, begin_methods, end_methods,
             num_head);

  return success();
}

template LogicalResult AttentionSliceMerge2<tpu::MatMulOp>::matchAndRewriteImpl(
    tpu::MatMulOp op, PatternRewriter &rewriter) const;

template LogicalResult
AttentionSliceMerge2<tpu::A16MatMulOp>::matchAndRewriteImpl(
    tpu::A16MatMulOp op, PatternRewriter &rewriter) const;

void sliceAttentionMerge2Split(PatternRewriter &rewriter, tpu::DevBeginOp op,
                               int64_t num_devices) {
  // Without StripIOQuant:
  // users: residual(norm), pos0, pos1, attn_mask, [past_k, past_v]
  // With StripIOQuant:
  // users: residual, norm, pos0, pos1, attn_mask, [past_k, past_v]
  std::vector<Operation *> users(op->user_begin(), op->user_end());
  bool decode_phase = (users.size() > 5);
  bool strip_io = users.size() == 5 || users.size() == 7;
  Operation *residual = users[0];
  Operation *attn_ln = users[strip_io];
  Operation *pos0 = users[strip_io + 1];
  Operation *pos1 = users[strip_io + 2];
  Operation *attn_mask = users[strip_io + 3];
  if (isa<tpu::CastOp>(users[0])) {
    std::vector<Operation *> use_ops(residual->user_begin(),
                                     residual->user_end());
    if (isa<tpu::LayerNormOp, tpu::RMSNormOp>(use_ops[0])) {
      attn_ln = use_ops[0];
    } else {
      attn_ln = use_ops[1];
    }
  }
  int num_head = op.getNumHead();

  std::vector<Value> end_operands;
  std::vector<Value> operands;
  Operation *end_op = nullptr;
  Operation *next_op;
  Value cur_out;
  std::vector<Value> other_opds;
  bool is_qwen_model = false;
  for (int cur_device = 0; cur_device < num_devices; ++cur_device) {
    auto suffix = std::to_string(cur_device);
    // clone residual branch
    next_op = residual;
    cur_out = next_op->getOperand(0);
    if (isa<tpu::AddOp>(next_op) &&
        cur_out.getDefiningOp() != attn_ln->getOperand(0).getDefiningOp()) {
      cur_out = next_op->getOperand(1);
    }
    if (isa<tpu::MatMulOp>(next_op)) {
      cur_out = next_op->getOperand(2);
    }
    if (isa<tpu::CastOp>(next_op)) {
      next_op = cloneCommonOp(rewriter, next_op, cur_out, -1, num_devices,
                              cur_device)[0];
    }
    auto ln_input = cur_out;
    createMulConstOp(rewriter, cur_out, cur_device, cur_device == 0 ? 1.0 : 0);
    auto residual_out = cur_out;

    // clone pos_ids input branch
    std::vector<Value> pos_ids;
    next_op = pos0;
    cur_out = next_op->getOperand(1);
    auto muls0 = cloneOpWithWeight(rewriter, next_op, cur_out, suffix);
    if (muls0.size() == 1 && isa<tpu::UnsqueezeOp>(muls0[0])) {
      next_op = muls0[0];
      if (isa<tpu::ReshapeOp, tpu::PermuteOp>(*next_op->user_begin())) {
        next_op = *next_op->user_begin();
      }
      createReshapeOp(rewriter, next_op, cur_out, cur_device);
      muls0 =
          std::vector<Operation *>(next_op->user_begin(), next_op->user_end());
    }
    pos_ids.push_back(cur_out);

    next_op = pos1;
    cur_out = next_op->getOperand(1);
    auto muls1 = cloneOpWithWeight(rewriter, next_op, cur_out, suffix);
    if (muls1.size() == 1 && isa<tpu::UnsqueezeOp>(muls1[0])) {
      next_op = muls1[0];
      if (isa<tpu::ReshapeOp, tpu::PermuteOp>(*next_op->user_begin())) {
        next_op = *next_op->user_begin();
      }
      createReshapeOp(rewriter, next_op, cur_out, cur_device);
      muls1 =
          std::vector<Operation *>(next_op->user_begin(), next_op->user_end());
    }
    pos_ids.push_back(cur_out);
    if (!isa<tpu::ConcatOp>(muls0[0]->getOperand(0).getDefiningOp())) {
      std::swap(pos_ids[0], pos_ids[1]);
    }

    // clone attn_mask input branch
    next_op = attn_mask;
    cur_out = next_op->getOperand(0);
    if (isa<tpu::MulConstOp>(cur_out.getDefiningOp())) {
      cur_out = next_op->getOperand(1);
    }
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

    // Attention
    next_op = attn_ln;
    cur_out = ln_input;
    std::vector<Operation *> op_branches =
        cloneOpWithWeight(rewriter, next_op, cur_out, suffix);

    // llama2: 0: value, 1: key, 2: query
    // qwen: 0: query, 1: key, 2: value
    if (cur_device == 0) {
      auto reshape = *op_branches[0]->getResult(0).user_begin();
      if (std::distance(reshape->user_begin(), reshape->user_end()) == 3) {
        is_qwen_model = true;
      }
    }

    Value attn_start_out = cur_out;
    // Value branch
    std::vector<Value> outs;
    outs.clear();
    next_op = is_qwen_model ? op_branches[2] : op_branches[0];
    cur_out = attn_start_out;
    other_opds = {past_v_out};
    next_op =
        cloneAttentionValue(rewriter, next_op, cur_out, other_opds, outs, 2,
                            num_devices, cur_device, false, num_head)[0];
    Value value_out = outs[0];
    Value value_branch = outs[1];
    // Query branch
    next_op = is_qwen_model ? op_branches[0] : op_branches[2];
    cur_out = attn_start_out;
    next_op = cloneAttentionQuery(rewriter, next_op, cur_out, pos_ids,
                                  num_devices, cur_device, false, num_head)[0];
    if (isa<tpu::MulConstOp>(next_op)) {
      next_op = cloneCommonOp(rewriter, next_op, cur_out, suffix);
    }
    Value query_branch = cur_out;
    auto query_next_op = next_op;
    // Key branch
    outs.clear();
    next_op = op_branches[1];
    cur_out = attn_start_out;
    other_opds = {pos_ids[0], pos_ids[1], past_k_out};
    next_op = cloneAttentionKey(rewriter, next_op, cur_out, other_opds, outs,
                                num_devices, cur_device, false, num_head)[0];
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

    // out0 = attn_out + residual_out
    if (!isa<tpu::AddOp>(next_op)) {
      auto matmul = attn_out.getDefiningOp();
      int num_opds = matmul->getNumOperands();
      matmul->setOperand(num_opds - 1, residual_out);
    } else {
      operands.clear();
      operands.push_back(residual_out);
      operands.push_back(attn_out);
      next_op = cloneMultiInsOp(rewriter, next_op, cur_out, operands, -1,
                                num_devices, cur_device);
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
  std::vector<Value> unused(end_op->operand_begin(), end_op->operand_end());
  end_op->setOperands(end_operands);

  module::removeUnusedOp();
}

LogicalResult
FAttentionSliceMerge::matchAndRewriteImpl(tpu::FAttentionOp op,
                                          PatternRewriter &rewriter) const {
  // TODO: only support quant input/output for now
  if (module::isOpInDevParallel(op) || !op->hasOneUse()) {
    return failure();
  }
  std::vector<Operation *> begin_ops;
  std::vector<Operation *> end_ops;
  // if fattention pattern is matched, no need to match GQA pattern
  // GQA is fixed to false
  bool GQA = false;
  int num_head = 0;
  if (!isFAttentionPattern(op, begin_ops, end_ops, GQA, &num_head)) {
    return failure();
  }
  // if kv_head < num_device, only support num_device % num_head == 0
  int num_device = module::getDeviceNum();
  if (num_head < num_device && num_device % num_head) {
    return failure();
  }
  std::vector<int64_t> begin_methods{1, 1, 1, 1, 1};
  if (begin_ops.size() > begin_methods.size()) {
    begin_methods.push_back(2); // past_k op
    begin_methods.push_back(2); // past_v op
  }
  std::vector<int64_t> end_methods{1, 3, 3};
  distribute(rewriter, begin_ops, end_ops,
             tpu::DevPattern::FAttentionSliceMerge, begin_methods, end_methods,
             num_head);
  return success();
}
void sliceFAttentionMergeSplit(PatternRewriter &rewriter, tpu::DevBeginOp op,
                               int64_t num_devices) {
  std::vector<Operation *> users(op->user_begin(), op->user_end());
  Operation *residual = users[0];
  Operation *attn_ln = users[1];
  std::vector<Operation *> pos = {users[2], users[3]};
  Operation *attn = users[4];
  int q_head = module::getShape(attn->getOperand(0))[2];
  int kv_head = op.getNumHead();

  std::vector<Value> end_operands;
  std::vector<Value> operands;
  Operation *end_op;
  Operation *next_op;
  Value cur_out;
  for (int cur_device = 0; cur_device < num_devices; ++cur_device) {
    auto suffix = std::to_string(cur_device);
    // clone residual branch
    next_op = residual;
    cur_out = next_op->getOperand(0);
    if (isa<tpu::AddOp>(next_op) &&
        cur_out.getDefiningOp() != op.getOperation()) {
      cur_out = next_op->getOperand(1);
    }
    if (!isa<tpu::MatMulOp>(next_op) && !isa<tpu::A16MatMulOp>(next_op)) {
      createMulConstOp(rewriter, cur_out, cur_device, 1. / num_devices);
    }
    Value residual_out = cur_out;

    // clone pos_ids input branch
    std::vector<Value> pos_ids;
    std::vector<Operation *> muls;
    for (int j = 0; j < 2; j++) {
      next_op = pos[j];
      cur_out = next_op->getOperand(1);
      auto next_ops = cloneOpWithWeight(rewriter, next_op, cur_out, suffix);
      if (next_ops.size() == 1 && isa<tpu::UnsqueezeOp>(next_ops[0])) {
        next_op = next_ops[0];
        if (isa<tpu::ReshapeOp, tpu::PermuteOp>(*next_op->user_begin())) {
          next_op = *next_op->user_begin();
        }
        createReshapeOp(rewriter, next_op, cur_out, cur_device);
        muls = std::vector<Operation *>(next_op->user_begin(),
                                        next_op->user_end());
      } else if (next_ops.size() == 2 && isa<tpu::MulOp>(next_ops[0])) {
        muls = next_ops;
      }
      pos_ids.push_back(cur_out);
    }
    if (isa<tpu::ConcatOp>(muls[0]->getOperand(0).getDefiningOp())) {
      std::swap(pos_ids[0], pos_ids[1]);
    }

    // clone past_k, past_v
    std::vector<Value> past_kv;
    if (users.size() > 5) {
      std::vector<Operation *> input_kv = {users[5], users[6]};
      for (int j = 0; j < 2; j++) {
        next_op = input_kv[j];
        cur_out = next_op->getOperand(0);
        next_op = createSliceOp(rewriter, next_op, cur_out, 2, num_devices,
                                cur_device, kv_head);
        past_kv.push_back(cur_out);
      }
    }

    // clone attention branch + out_proj + residual_add
    next_op = attn_ln;
    cur_out = next_op->getOperand(0);
    next_op = cloneFlashAttention(rewriter, next_op, cur_out, pos_ids, past_kv,
                                  num_devices, cur_device, q_head, kv_head)[0];
    next_op = cloneRowParallelMatMul(rewriter, next_op, cur_out, num_devices,
                                     cur_device, q_head);
    if (isa<tpu::AddOp>(residual)) {
      operands = {residual_out, cur_out};
      next_op = cloneMultiInsOp(rewriter, next_op, cur_out, operands, -1,
                                num_devices, cur_device);
    }
    end_operands.push_back(cur_out);
    end_operands.push_back(past_kv[0]);
    end_operands.push_back(past_kv[1]);
    if (cur_device == 0) {
      end_op = next_op;
    } else {
      assert(end_op == next_op);
    }
  }

  assert(isa<tpu::DevEndOp>(end_op));
  std::vector<Value> unused(end_op->operand_begin(), end_op->operand_end());
  end_op->setOperands(end_operands);

  module::removeUnusedOp();
}

LogicalResult
EmbeddingSliceMerge::matchAndRewriteImpl(tpu::GatherOp op,
                                         PatternRewriter &rewriter) const {
  if (module::isOpInDevParallel(op)) {
    return failure();
  }
  std::vector<Operation *> begin_ops{op};
  std::vector<Operation *> end_ops{op};
  std::vector<int64_t> begin_methods{1};
  std::vector<int64_t> end_methods{1};

  distribute(rewriter, begin_ops, end_ops, tpu::DevPattern::EmbeddingSliceMerge,
             begin_methods, end_methods, 0);
  return success();
}

void embeddingMergeSplit(PatternRewriter &rewriter, tpu::DevBeginOp op,
                         int64_t num_devices) {
  auto gather_op = *op->getResult(0).user_begin();
  if (auto gather = dyn_cast<tpu::GatherOp>(gather_op)) {
    gather.setIfNegIndex(false);
  }
  auto src_op = gather_op->getOperand(0).getDefiningOp();
  auto weight_op = dyn_cast_or_null<top::WeightOp>(src_op);
  auto vocab_size = module::getShape(weight_op.getOutput())[0];
  auto length = ceiling_func(vocab_size, num_devices);

  std::vector<Value> end_operands;
  Operation *end_op = nullptr;
  Operation *next_op;
  Value cur_out;
  for (int cur_device = 0; cur_device < num_devices; ++cur_device) {
    next_op = gather_op;
    cur_out = next_op->getOperand(1);
    if (cur_device) {
      float sub_val = cur_device * length;
      createSubConstOp(rewriter, cur_out, cur_device, sub_val);
    }
    next_op = cloneOpWithWeight(rewriter, next_op, cur_out, 0, num_devices,
                                cur_device)[0];
    end_operands.push_back(cur_out);
    if (cur_device == 0) {
      end_op = next_op;
    } else {
      assert(end_op == next_op);
    }
  }
  assert(isa<tpu::DevEndOp>(end_op));
  std::vector<Value> unused(end_op->operand_begin(), end_op->operand_end());
  end_op->setOperands(end_operands);

  module::removeUnusedOp();
}

} // namespace tpu
} // namespace tpu_mlir
