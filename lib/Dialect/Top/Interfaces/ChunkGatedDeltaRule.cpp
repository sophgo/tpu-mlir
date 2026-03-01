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

int64_t top::ChunkGatedDeltaRuleOp::getFLOPs() {
  // query/key/value shape: [B, S, H, D] (before internal transpose to [B, H, S,
  // D]) recurrent_state shape: [B, H, D, D]
  auto v_shape = module::getShape(getValue());
  int64_t B = v_shape[0];
  int64_t S = v_shape[1];
  int64_t H = v_shape[2];
  int64_t D = v_shape[3];
  int64_t cs = getChunkSize();
  // pad S to be divisible by chunk_size
  int64_t S_pad = ((S + cs - 1) / cs) * cs;
  int64_t nc = S_pad / cs;

  // MatMul FLOPs (each matmul of [M,K]*[K,N] = 2*M*K*N):
  //
  // Non-loop matmuls (over all chunks):
  // 1. k_beta @ key^T:          (B,H,nc,cs,D) x (B,H,nc,D,cs) =>
  // B*H*nc*cs^2*D*2
  // 2. attn @ v_beta:            (B,H,nc,cs,cs) x (B,H,nc,cs,D) =>
  // B*H*nc*cs^2*D*2
  // 3. attn @ (k_beta*g_exp):    (B,H,nc,cs,cs) x (B,H,nc,cs,D) =>
  // B*H*nc*cs^2*D*2
  // 4. query @ key^T:            (B,H,nc,cs,D) x (B,H,nc,D,cs) =>
  // B*H*nc*cs^2*D*2
  //
  // Loop matmuls (nc iterations, summed):
  // 5. k_cumdecay_i @ state:     (B,H,cs,D) x (B,H,D,D) => B*H*nc*cs*D^2*2
  // 6. q_g_i @ state:            (B,H,cs,D) x (B,H,D,D) => B*H*nc*cs*D^2*2
  // 7. intra_chunk_attn_i @ v_new:(B,H,cs,cs) x (B,H,cs,D) => B*H*nc*cs^2*D*2
  // 8. k_g_diff_t_i @ v_new:     (B,H,D,cs) x (B,H,cs,D) => B*H*nc*cs*D^2*2

  int64_t common = 2 * B * H * nc;
  // #1 + #2 + #3 + #4 + #7: 5 * cs^2 * D
  int64_t chunk_matmul_flops = common * cs * cs * D * 5;
  // #5 + #6 + #8: 3 * cs * D^2
  int64_t recurrent_matmul_flops = common * cs * D * D * 3;

  return chunk_matmul_flops + recurrent_matmul_flops;
}

LogicalResult top::ChunkGatedDeltaRuleOp::init(InferenceParameter &p) {
  return success();
}
void top::ChunkGatedDeltaRuleOp::deinit(InferenceParameter &p) {}

LogicalResult top::ChunkGatedDeltaRuleOp::inference(InferenceParameter &p) {
  UNREACHABLE_THIS("Not Implemented");
  return success();
}

// if keep_dims, output shape = input shape
// else input = [1, M_q, q_head, d], output = [1, M_q, q_head*d]
void top::ChunkGatedDeltaRuleOp::shape_inference() {
  auto value = getValue();
  auto recurrent_state = getRecurrentState();
  auto v_shape = module::getShape(value);
  auto r_shape = module::getShape(recurrent_state);
  module::setShapeOrVerify(getAttnOut(), v_shape);
  module::setShapeOrVerify(getNewRecurrentState(), r_shape);
}
