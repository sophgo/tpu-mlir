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
// pattern MatMulSliceMerge
// e.g. ChatGlm2
// ======================================

namespace tpu_mlir {
namespace tpu {

// e.g. [12, 16, 18] => [12, 16, 9]
static bool isHalfSlice(tpu::SliceOp op) {
  auto offset = module::getI64Array(op.getOffset());
  auto steps = module::getI64Array(op.getSteps());
  auto in_shape = module::getShape(op.getInput());
  auto out_shape = module::getShape(op.getOutput());
  for (int i = 0; i < in_shape.size(); i++) {
    if (steps->at(i) != 1) {
      return false;
    }
    auto o = offset->at(i);
    if (i == in_shape.size() - 1) {
      if ((o != 0 && o != in_shape[i] / 2) || out_shape[i] != in_shape[i] / 2) {
        return false;
      }
    } else if (o != 0 || in_shape[i] != out_shape[i]) {
      return false;
    }
  }
  return true;
}

template <typename MatMulTy>
LogicalResult MatMulSliceMerge<MatMulTy>::matchAndRewriteImpl(
    MatMulTy op, PatternRewriter &rewriter) const {
  if (!isLargeMatMul(op) || module::isOpInDevParallel(op)) {
    return failure();
  }
  if (!module::isNone(op.getBias()) && !module::isWeight(op.getBias())) {
    return failure();
  }
  std::vector<Operation *> users(op->user_begin(), op->user_end());
  if (users.size() != 2) {
    return failure();
  }
  Operation *res_op = nullptr;
  for (auto user : users) {
    auto slice = dyn_cast<tpu::SliceOp>(user);
    if (!slice || !slice->hasOneUse() || !isHalfSlice(slice)) {
      return failure();
    }
    auto next = *slice->user_begin();
    while (next != nullptr) {
      if (isBinaryOp(next)) {
        if (res_op == nullptr) {
          res_op = next;
          continue;
        } else if (next != res_op) {
          return failure();
        }
        break;
      } else if (false == next->hasOneUse() ||
                 !next->hasTrait<trait::SupportElementwise>()) {
        return failure();
      }
      next = *next->user_begin();
    }
  }
  if (!res_op->hasOneUse()) {
    return failure();
  }
  auto next = *res_op->user_begin();
  while (next != nullptr) {
    if (isLargeMatMul(next)) {
      break;
    }
    if (false == next->hasOneUse() ||
        !next->hasTrait<trait::SupportElementwise>()) {
      return failure();
    }
  }
  // Bingo !!
  distribute(rewriter, op, next, tpu::DevPattern::MatMulSliceMerge);
  return success();
}

template LogicalResult MatMulSliceMerge<tpu::MatMulOp>::matchAndRewriteImpl(
    tpu::MatMulOp op, PatternRewriter &rewriter) const;

template LogicalResult MatMulSliceMerge<tpu::A16MatMulOp>::matchAndRewriteImpl(
    tpu::A16MatMulOp op, PatternRewriter &rewriter) const;

template <typename MatMulTy>
void sliceMergeSplit(MatMulTy mm0, PatternRewriter &rewriter,
                     tpu::DevBeginOp op, int64_t num_devices) {
  if (!mm0) {
    return;
  }
  auto next_op = *op->user_begin();
  auto filterOp = mm0.getOperand(1).template getDefiningOp<top::WeightOp>();
  auto filterShape = module::getShape(filterOp.getOutput());
  auto outputShape = module::getShape(mm0.getOutput());
  std::vector<NamedAttribute> attrs(op->getAttrs().begin(),
                                    op->getAttrs().end());
  auto has_bias = !module::isNone(mm0.getBias());
  auto num_dims = filterShape.size();
  // weight shape: (KxN)
  auto N = filterShape[num_dims - 1];
  auto N_half = N / 2;
  auto a16_mm0 = dyn_cast<tpu::A16MatMulOp>(mm0.getOperation());
  if (a16_mm0) {
    std::vector<NamedAttribute> mm0_attrs(mm0->getAttrs().begin(),
                                          mm0->getAttrs().end());
    attrs.insert(attrs.end(), mm0_attrs.begin(), mm0_attrs.end());
  }
  int mm0_wbits = a16_mm0 ? a16_mm0.getWeightBits() : 16;
  auto slice_n = ceiling_func(N_half, num_devices);
  std::vector<Operation *> slices(mm0->user_begin(), mm0->user_end());
  auto slice0Op = cast<tpu::SliceOp>(slices[0]);
  auto offset = module::getI64Array(slice0Op.getOffset());
  if (offset->back() != 0) {
    std::swap(slices[0], slices[1]);
  }
  std::vector<Value> end_operands;
  Operation *end_op = nullptr;
  bool biasAdd = false;
  Value biasValue;
  for (int i = 0; i < num_devices; i++) {
    std::vector<Value> res_operands;
    auto offset = i * slice_n;
    auto length = std::min(slice_n, N_half - offset);
    auto suffix = std::to_string(i);
    // slice one half
    for (int half = 0; half < 2; half++) {
      auto offset_half = offset + half * N_half;
      auto suffix_half = suffix + "_" + std::to_string(half);
      auto newFilter0 = module::opSliceAxis(rewriter, mm0.getOperand(1),
                                            num_dims - 1, offset_half, length);
      std::vector<Value> operands;
      operands.push_back(mm0.getInput());
      operands.push_back(newFilter0);
      if (a16_mm0) {
        auto scale = mm0.getOperand(2).template getDefiningOp<top::WeightOp>();
        operands.push_back(scale.clone(suffix_half));
      }
      if (has_bias) {
        auto new_bias = module::opSliceAxis(rewriter, mm0.getBias(),
                                            num_dims - 1, offset_half, length);
        operands.push_back(new_bias);
      } else {
        operands.push_back(mm0.getBias());
      }
      auto new_loc = module::getLocLike(mm0.getOutput(), suffix_half);
      std::vector<int64_t> new_shape = outputShape;
      new_shape[new_shape.size() - 1] = (mm0_wbits == 4 ? 2 : 1) * length;
      auto new_type = module::getTypeLike(mm0.getOutput(), new_shape);
      rewriter.setInsertionPointAfter(mm0);
      auto new_mm0 =
          rewriter.create<MatMulTy>(new_loc, new_type, operands, attrs);
      Value cur_output = new_mm0.getOutput();
      next_op = *slices[half]->user_begin();
      while (!isBinaryOp(next_op)) {
        auto new_op = cloneOp(rewriter, next_op, new_shape, suffix_half);
        new_op->setOperand(0, cur_output);
        cur_output = new_op->getResult(0);
        next_op = *next_op->user_begin();
      }
      res_operands.push_back(cur_output);
    }
    // res_op: add/mul
    auto new_shape = module::getShape(res_operands[0]);
    auto new_op = cloneOp(rewriter, next_op, new_shape, suffix);
    new_op->setOperands(res_operands);
    Value cur_output = new_op->getResult(0);
    next_op = *next_op->user_begin();
    // matmul op, can be MatMul or A16MatMul
    while (!isa<tpu::MatMulOp, tpu::A16MatMulOp>(next_op)) {
      new_op = cloneOp(rewriter, next_op, new_shape, suffix);
      new_op->setOperand(0, cur_output);
      cur_output = new_op->getResult(0);
      next_op = *next_op->user_begin();
    }
    auto a16_mm1 = dyn_cast<tpu::A16MatMulOp>(next_op);
    auto mm1_weight_value = next_op->getOperand(1);
    auto mm1_bias_value = next_op->getOperand(a16_mm1 ? 3 : 2);

    auto new_loc = module::getLocLike(next_op, suffix);
    std::vector<Value> operands;
    operands.push_back(cur_output);
    auto newFilter1 = module::opSliceAxis(
        rewriter, mm1_weight_value, num_dims - 2,
        (mm0_wbits == 4 ? 2 : 1) * offset, (mm0_wbits == 4 ? 2 : 1) * length);
    operands.push_back(newFilter1);
    if (a16_mm1) {
      auto new_scale = module::opSliceAxis(rewriter, a16_mm1.getOperand(2), 0,
                                           (mm0_wbits == 4 ? 2 : 1) * offset,
                                           (mm0_wbits == 4 ? 2 : 1) * length);
      operands.push_back(new_scale);
    }
    if (module::isNone(mm1_bias_value)) {
      operands.push_back(mm1_bias_value);
    } else if (module::isWeight(mm1_bias_value)) {
      auto bias = mm1_bias_value.template getDefiningOp<top::WeightOp>();
      operands.push_back(bias.clone(suffix));
    } else {
      operands.push_back(module::getNoneOp(op));
      biasAdd = true;
      biasValue = mm1_bias_value;
    }
    rewriter.setInsertionPointAfter(next_op);
    auto new_type = next_op->getResult(0).getType();
    mlir::Operation *new_mm1;

    auto createMatMulOp = [&](bool a16_mm1) -> mlir::Operation * {
      if (a16_mm1) {
        return rewriter.create<tpu::A16MatMulOp>(new_loc, new_type, operands,
                                                 next_op->getAttrs());
      } else {
        operands.push_back(module::getNoneOp(op));
        operands.push_back(module::getNoneOp(op));
        return rewriter.create<tpu::MatMulOp>(new_loc, new_type, operands,
                                              next_op->getAttrs());
      }
    };

    new_mm1 = createMatMulOp(a16_mm1);

    if (i == 0) {
      end_op = *next_op->user_begin();
    } else {
      assert(end_op == *next_op->user_begin());
    }
  }
  assert(isa<tpu::DevEndOp>(end_op));
  end_op->setOperands(end_operands);
  if (biasAdd) {
    auto dis_op = cast<tpu::DevEndOp>(end_op);
    auto dis_out = dis_op.getOutputs()[0];
    rewriter.setInsertionPointAfter(end_op);
    auto new_loc = module::getLocLike(dis_out, "add");
    auto add_op = rewriter.create<tpu::AddOp>(new_loc, dis_out.getType(),
                                              ValueRange{dis_out, biasValue});
    dis_out.replaceAllUsesExcept(add_op.getOutput(), add_op);
  }
  eraseForward(rewriter, mm0);
}

template void sliceMergeSplit(tpu::MatMulOp mm0, PatternRewriter &rewriter,
                              tpu::DevBeginOp op, int64_t num_devices);

template void sliceMergeSplit(tpu::A16MatMulOp mm0, PatternRewriter &rewriter,
                              tpu::DevBeginOp op, int64_t num_devices);

/**
 * Attention Tensor Parallelism
 */
template <typename MatMulTy>
LogicalResult AttentionSliceMerge<MatMulTy>::matchAndRewriteImpl(
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
  if (!isChatGLMAttentionPattern(op, attn_begin_ops, attn_end_ops, &num_head)) {
    return failure();
  }

  auto attn_top_op = attn_begin_ops[0];
  bool has_cast = false;
  if (isa<tpu::CastOp>(attn_top_op->getOperand(0).getDefiningOp())) {
    attn_top_op = attn_top_op->getOperand(0).getDefiningOp();
    attn_begin_ops[0] = attn_top_op;
    has_cast = true;
  }

  // residual, attn_ln, pos_id, attn_mask op
  std::vector<Operation *> begin_ops;
  if (!has_cast) {
    begin_ops.push_back(begin_op);
  }
  begin_ops.insert(begin_ops.end(), attn_begin_ops.begin(),
                   attn_begin_ops.end());
  std::vector<int64_t> begin_methods{1, 1, 1};
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
  distribute(rewriter, begin_ops, end_ops, tpu::DevPattern::AttentionSliceMerge,
             begin_methods, end_methods, num_head);

  return success();
}

template LogicalResult AttentionSliceMerge<tpu::MatMulOp>::matchAndRewriteImpl(
    tpu::MatMulOp op, PatternRewriter &rewriter) const;

template LogicalResult
AttentionSliceMerge<tpu::A16MatMulOp>::matchAndRewriteImpl(
    tpu::A16MatMulOp op, PatternRewriter &rewriter) const;

void sliceAttentionMergeSplit(PatternRewriter &rewriter, tpu::DevBeginOp op,
                              int64_t num_devices) {
  // Without StripIOQuant:
  // users: residual(norm), pos, attn_mask, [past_k, past_v]
  // With StripIOQuant:
  // users: residual, norm, pos, attn_mask, [past_k, past_v]
  // block quant=false 3
  // block quant=true 4
  // block_cache quant=false 5
  // block_cache quant=true 6
  std::vector<Operation *> users(op->user_begin(), op->user_end());
  bool decode_phase = (users.size() > 4);
  bool strip_io = users.size() == 4 || users.size() == 6;
  Operation *residual = users[0];
  Operation *attn_ln = users[strip_io];
  Operation *pos = users[strip_io + 1];
  Operation *attn_mask = users[strip_io + 2];
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

  std::vector<Value> operands;
  std::vector<Value> end_operands;
  std::vector<Value> pos_operands;
  Operation *end_op = nullptr;
  Operation *next_op;
  std::vector<Operation *> next_ops;
  Value cur_out;
  std::vector<Value> other_opds;
  for (int cur_device = 0; cur_device < num_devices; ++cur_device) {
    auto suffix = std::to_string(cur_device);
    // clone residual branch
    next_op = residual;
    if (isa<tpu::MatMulOp>(next_op)) {
      cur_out = next_op->getOperand(2);
    } else {
      cur_out = next_op->getOperand(0);
    }
    if (!isa<tpu::CastOp, tpu::DevBeginOp>(cur_out.getDefiningOp())) {
      llvm_unreachable("This Op should be CastOp or DevBeginOp.\n");
    }
    if (isa<tpu::CastOp>(next_op)) {
      next_op = cloneCommonOp(rewriter, next_op, cur_out, -1, num_devices,
                              cur_device)[0];
    }
    auto ln_input = cur_out;
    // createMulConstOp(rewriter, cur_out, cur_device, cur_device == 0 ? 1.0 :
    // 0);
    auto residual_out = cur_out;

    // clone pos_ids input branch
    next_op = pos;
    cur_out = next_op->getOperand(1);
    pos_operands = cloneChatGLMPosInput(rewriter, next_op, cur_out, -1,
                                        num_devices, cur_device, suffix);

    // clone attn_mask input branch
    next_op = attn_mask;
    if (isa<tpu::DevBeginOp>(next_op->getOperand(0).getDefiningOp())) {
      cur_out = next_op->getOperand(0);
    } else {
      cur_out = next_op->getOperand(1);
    }
    // (Cast) -> Reshape
    while (isa<tpu::CastOp, tpu::MulConstOp>(next_op)) {
      next_op = cloneCommonOp(rewriter, next_op, cur_out, -1, num_devices,
                              cur_device)[0];
    }
    auto attn_mask_out = cur_out;

    Value past_k_out, past_v_out;
    // not need to split kvcache
    if (decode_phase) {
      // clone past_k, past_v
      for (int j = users.size() - 2; j < users.size(); j++) {
        next_op = users[j];
        cur_out = next_op->getOperand(0);
        if (isa<tpu::CastOp>(next_op)) {
          next_op = cloneCommonOp(rewriter, next_op, cur_out, -1, num_devices,
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
    Value attn_start_out = cur_out;
    // Value branch
    std::vector<Value> outs;
    outs.clear();
    next_op = op_branches[2];
    cur_out = attn_start_out;
    other_opds = {past_v_out};
    auto value_next_op =
        cloneChatGLMAttentionValue(rewriter, next_op, cur_out, other_opds, outs,
                                   -1, num_devices, cur_device, num_head)[0];
    Value value_out = outs[0];
    Value value_branch = outs[1];
    // Query branch
    next_op = op_branches[0];
    cur_out = attn_start_out;
    auto query_next_ops =
        cloneChatGLMAttentionQK(rewriter, next_op, cur_out, 2, pos_operands,
                                num_devices, cur_device, num_head);
    Value query_out = cur_out;
    // Key branch
    outs.clear();
    next_op = op_branches[1];
    cur_out = attn_start_out;
    auto key_next_ops =
        cloneChatGLMAttentionQK(rewriter, next_op, cur_out, -1, pos_operands,
                                num_devices, cur_device, num_head);
    Value key_out = cur_out;
    // Q@K
    operands.clear();
    operands.push_back(query_out);
    operands.push_back(key_out);
    if (past_k_out != 0x0)
      operands.push_back(past_k_out);
    next_op = cloneChatGLMAttentionQxK(rewriter, query_next_ops, key_next_ops,
                                       next_op, cur_out, operands, num_devices,
                                       cur_device)[0];
    // Attention Matrix branch
    other_opds = {attn_mask_out};
    auto qk_next_op = cloneAttentionMatrix(
        rewriter, next_op, cur_out, 1, other_opds, num_devices, cur_device)[0];
    Value qk_out = cur_out;
    // QK@V
    next_op = cloneChatGLMAttentionOutput(
        rewriter, qk_next_op, value_next_op, next_op, value_branch, qk_out,
        cur_out, num_devices, cur_device, num_head);
    Value attn_out = cur_out;

    // out0 = attn_out + residual_out
    if (!isa<tpu::AddOp>(next_op)) {
      auto matmul = attn_out.getDefiningOp();
      createMulConstOp(rewriter, residual_out, cur_device, 1.0 / num_devices);
      matmul->setOperand(2, residual_out);
    } else {
      createMulConstOp(rewriter, residual_out, cur_device, 1.0 / num_devices);

      operands.clear();
      operands.push_back(residual_out);
      operands.push_back(attn_out);
      next_op = cloneMultiInsOp(rewriter, next_op, cur_out, operands, -1,
                                num_devices, cur_device);
    }

    // postprocess for kv
    for (auto user : key_next_ops) {
      if (isa<tpu::CastOp>(user)) {
        cloneCommonOp(rewriter, user, key_out, suffix);
      }
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

} // namespace tpu
} // namespace tpu_mlir
