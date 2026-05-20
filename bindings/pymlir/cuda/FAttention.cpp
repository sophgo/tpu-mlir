//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "../pycuda.h"
#include "cuda_helper.h"

void py_cuda::cudaFAttentionOp(top::FAttentionOp op) {
  int batch = op.getBatch(), M_q = op.getMq(), M_k = op.getMk();
  int q_head = op.getQHead(), kv_head = op.getKvHead(), d = op.getDim();
  (void)kv_head; // GQA support pending;
  float scale = (float)op.getScale().convertToDouble();
  int Hd = q_head * d;

  // Step 1: Q@K^T / scale → [batch, q_head, M_q, M_k]
  auto scores = cuda_malloc(batch * q_head * M_q * M_k * sizeof(float));
  cuda::bmAttentionQK(getCudaData(op.getQueries()), getCudaData(op.getKeys()),
                       scores.get(), batch, q_head, M_q, M_k, d, scale);

  // Step 2: mask (optional) — broadcast [M_q, M_k] to [batch, q_head, M_q, M_k]
  // TODO: implement mask addition

  // Step 3: softmax along M_k
  cuda::bmSoftmax(scores.get(), nullptr, scores.get(),
                   batch * q_head * M_q, M_k, 1, false);

  // Step 4: scores@V → [batch, q_head, M_q, d]
  auto context = cuda_malloc(batch * q_head * M_q * d * sizeof(float));
  cuda::bmAttentionPV(scores.get(), getCudaData(op.getValues()),
                       context.get(), batch, q_head, M_q, M_k, d);
  scores.reset();

  // Step 5: transpose [batch, q_head, M_q, d] → [batch, M_q, q_head, d]
  auto ctx_perm = cuda_malloc(batch * q_head * M_q * d * sizeof(float));
  cuda::bmPermuteBMHD(context.get(), ctx_perm.get(), batch, q_head, M_q, d);
  context.reset();

  // Step 6: reshape to [batch, M_q, q_head*d] and copy to output
  CHECK_CUDA(cudaMemcpy(getCudaData(op.getOutput()), ctx_perm.get(),
                        batch * M_q * Hd * sizeof(float),
                        cudaMemcpyDeviceToDevice));
}
